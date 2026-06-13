// MainWindow.xaml.cs — three buttons, one robot, zero menus.
// The cartoon robot's walk quality is driven by the backend's skill value:
// low skill = wobbling and falling over, high skill = confident strides.
using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using System.Windows.Shapes;
using System.Windows.Threading;
using IOPath = System.IO.Path;

namespace Aistank.Editor;

public partial class MainWindow : Window
{
    private readonly IRobotBackend _backend;
    private readonly DispatcherTimer _timer;
    private readonly Random _rng = new();

    private double _clock;            // animation time, only advances when moving
    private double _fallTimer;        // > 0 while the robot is on the ground
    private double _groundScroll;
    private string _lastMessage = "";

    // 3D viewport (real engine): a colored crowd of the first K agents.
    private const int CrowdSize = 12;
    private const double CrowdSpacing = 1.6;
    private Scene3D? _scene;
    private double _camAz = 0.0, _camEl = 0.30, _camDist = 9.0;
    private System.Windows.Point _lastMouse;
    private bool _dragging;

    public MainWindow()
    {
        InitializeComponent();

        _backend = CreateBackend();
        if (_backend.IsReal)
        {
            Title = "Robot School — real MuJoCo physics (3D)";
            if (_backend.Geoms is { } geoms)
            {
                _scene = new Scene3D(SceneRoot, geoms, (int)geoms.Count, CrowdSize, CrowdSpacing);
                Stage.Visibility = Visibility.Collapsed;   // hide 2D canvas
                UpdateCamera();
            }
        }
        else
        {
            Title = "Robot School — demo mode";
            View3D.Visibility = Visibility.Collapsed;       // hide 3D, use 2D cartoon
            DemoBanner.Visibility = Visibility.Visible;
        }

        _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(16) };
        _timer.Tick += (_, _) => Frame();
        _timer.Start();

        SpeechBubble.Text = "Hi! Press TRAIN to teach me to walk!";
        Closed += (_, _) => _backend.Dispose();
    }

    private static IRobotBackend CreateBackend()
    {
        try
        {
            return new NativeBackend(IOPath.Combine(AppContext.BaseDirectory,
                                                    "assets", "humanoid16.xml"));
        }
        catch (Exception)
        {
            // Native engine not built (or no DX12 GPU / MuJoCo missing) — stay playable.
            return new DemoBackend();
        }
    }

    // ----------------------------------------------------------- Buttons

    private void OnTrain(object sender, RoutedEventArgs e)
    {
        _backend.StartTraining();
        SetButtons(running: true);
    }

    private void OnPlay(object sender, RoutedEventArgs e)
    {
        _backend.StartPlaying();
        SetButtons(running: true);
    }

    private void OnStop(object sender, RoutedEventArgs e)
    {
        _backend.Stop();
        SetButtons(running: false);
        SpeechBubble.Text = "Taking a break! 😴";
        _lastMessage = SpeechBubble.Text;
    }

    private void SetButtons(bool running)
    {
        TrainButton.IsEnabled = !running;
        PlayButton.IsEnabled = !running;
        StopButton.IsEnabled = running;
    }

    // ----------------------------------------------------------- Frame loop

    private void Frame()
    {
        var status = _backend.GetStatus();
        const double dt = 0.016;

        UpdateSkillMeter(status);
        UpdateSpeech(status);
        if (_scene != null) _scene.Update(_backend);   // real 3D crowd
        else DrawScene(status, dt);                     // 2D demo fallback
    }

    // ----------------------------------------------------------- 3D camera

    private void UpdateCamera()
    {
        var target = new Point3D(0, 0, 0.8);
        double ce = Math.Cos(_camEl), se = Math.Sin(_camEl);
        var pos = new Point3D(
            target.X + _camDist * ce * Math.Sin(_camAz),
            target.Y - _camDist * ce * Math.Cos(_camAz),
            target.Z + _camDist * se);
        Cam3D.Position = pos;
        Cam3D.LookDirection = new Vector3D(target.X - pos.X, target.Y - pos.Y, target.Z - pos.Z);
    }

    private void View3D_MouseDown(object sender, MouseButtonEventArgs e)
    {
        _dragging = true; _lastMouse = e.GetPosition(View3D); View3D.CaptureMouse();
    }
    private void View3D_MouseUp(object sender, MouseButtonEventArgs e)
    {
        _dragging = false; View3D.ReleaseMouseCapture();
    }
    private void View3D_MouseMove(object sender, MouseEventArgs e)
    {
        if (!_dragging) return;
        var p = e.GetPosition(View3D);
        _camAz += (p.X - _lastMouse.X) * 0.01;
        _camEl = Math.Clamp(_camEl - (p.Y - _lastMouse.Y) * 0.01, -0.5, 1.3);
        _lastMouse = p;
        UpdateCamera();
    }
    private void View3D_MouseWheel(object sender, MouseWheelEventArgs e)
    {
        _camDist = Math.Clamp(_camDist * (1 - e.Delta * 0.0008), 3.0, 25.0);
        UpdateCamera();
    }

    private void UpdateSkillMeter(RobotStatus status)
    {
        SkillFill.Width = 320 * status.Skill;
        int stars = (int)Math.Round(status.Skill * 5);
        StarsText.Text = new string('⭐', stars) + new string('☆', 5 - stars);
        StarsText.Foreground = Brushes.Gold;
    }

    private void UpdateSpeech(RobotStatus status)
    {
        string msg = status.Mode switch
        {
            RobotMode.Idle => _lastMessage,
            RobotMode.Playing => "Watch me go! 🎉",
            RobotMode.Training when status.Skill < 0.2f => "Whoa! Walking is HARD! 🍮",
            RobotMode.Training when status.Skill < 0.5f => "Baby steps... baby steps... 👣",
            RobotMode.Training when status.Skill < 0.8f => "Hey, I'm getting it! 💪",
            RobotMode.Training => "Look at me, I'm a walking champion! 🏆",
            _ => "",
        };
        if (msg.Length > 0 && msg != SpeechBubble.Text)
        {
            SpeechBubble.Text = msg;
            _lastMessage = msg;
        }
    }

    // ----------------------------------------------------------- Scene dispatch

    private float[]? _geomXpos, _geomXmat;

    private void DrawScene(RobotStatus status, double dt)
    {
        Stage.Children.Clear();
        double w = Stage.ActualWidth, h = Stage.ActualHeight;
        if (w < 50 || h < 50) return;

        // Real engine → draw agent 0's actual MuJoCo collision geometry, with
        // full 3D orientation, so it visibly articulates and topples like a body.
        if (_backend.Geoms is { } geoms)
        {
            int n = geoms.Count;
            _geomXpos ??= new float[n * 3];
            _geomXmat ??= new float[n * 9];
            if (_backend.TryGetGeomPose(_geomXpos, _geomXmat))
            {
                DrawPhysicsRobot(geoms, _geomXpos, _geomXmat, w, h);
                return;
            }
        }

        double groundY = h * 0.78;
        bool moving = status.Mode != RobotMode.Idle && _fallTimer <= 0;
        double speed = 60 + 180 * status.Skill;

        if (moving)
        {
            _clock += dt * (2.0 + 4.0 * status.Skill);
            _groundScroll = (_groundScroll + speed * dt) % 60;

            // Low skill: robot randomly tumbles. At skill 1 it (almost) never falls.
            double fallChance = status.Mode == RobotMode.Playing
                ? 0.0005 : (1.0 - status.Skill) * 0.012;
            if (_rng.NextDouble() < fallChance) _fallTimer = 1.2;
        }
        if (_fallTimer > 0) _fallTimer -= dt;

        // Scrolling ground stripes sell the walking motion.
        for (double x = -_groundScroll; x < w; x += 60)
            Stage.Children.Add(MakeLine(x, groundY + 14, x + 30, groundY + 14,
                                        Color.FromRgb(0x4e, 0x83, 0x3f), 4));

        DrawRobot(w * 0.45, groundY, status.Skill, fallen: _fallTimer > 0);

        if (_fallTimer > 0.9) SpeechBubble.Text = "Oops! 💥";
    }

    // MuJoCo geom type enums we render.
    private const int GeomPlane = 0, GeomSphere = 2, GeomCapsule = 3, GeomCylinder = 5, GeomBox = 6;

    // Camera: side view down world -Y, with a gentle oblique tilt so lateral
    // offset (left/right legs, depth) reads as 3D. World x = forward → screen x
    // (camera tracks the robot), world z = up → screen y. The same real MuJoCo
    // xpos/xmat that drive the physics drive every shape here — nothing is faked.
    private void DrawPhysicsRobot(GeomModel geoms, float[] xpos, float[] xmat, double w, double h)
    {
        const double scale = 230;                 // pixels per metre
        double groundY = h * 0.80;
        int n = geoms.Count;

        // 3/4 camera: yaw 32° around the vertical axis so left/right limbs spread
        // apart and the robot reads as a 3D character instead of an edge-on stack.
        // fx (screen horizontal) mixes forward+lateral; depth drives shading and a
        // slight vertical parallax. Walking (+x) still moves the robot rightward.
        const double yaw = 0.56;                  // ~32°
        double cosY = Math.Cos(yaw), sinY = Math.Sin(yaw);

        double Fx(double wx, double wy) => wx * cosY + wy * sinY;
        double Depth(double wx, double wy) => -wx * sinY + wy * cosY;

        // Track the robot horizontally by the mean rotated-x of its dynamic geoms.
        double cfx = 0; int dyn = 0;
        for (int g = 0; g < n; g++)
            if (geoms.Types[g] != GeomPlane) { cfx += Fx(xpos[g * 3 + 0], xpos[g * 3 + 1]); dyn++; }
        if (dyn > 0) cfx /= dyn;
        double camX = w * 0.42;

        (double X, double Y) Project(double wx, double wy, double wz) =>
            (camX + (Fx(wx, wy) - cfx) * scale,
             groundY - wz * scale + Depth(wx, wy) * 0.18 * scale);

        // --- Sky already painted by the viewport gradient; draw ground + grid ---
        Stage.Children.Add(new Rectangle
        {
            Width = w, Height = Math.Max(0, h - groundY) + 4,
            Fill = new SolidColorBrush(Color.FromRgb(0x5E, 0x9C, 0x4C)),
        }.At(0, groundY));
        double scroll = ((cfx * scale) % 64 + 64) % 64;
        for (double x = -scroll; x < w; x += 64)
            Stage.Children.Add(MakeLine(x, groundY, x + 22, groundY + 18,
                                        Color.FromRgb(0x4c, 0x82, 0x3d), 3));
        Stage.Children.Add(MakeLine(0, groundY, w, groundY, Color.FromRgb(0x3c, 0x66, 0x30), 3));

        // --- Soft contact shadows: each geom projected straight down to z = 0 ---
        for (int g = 0; g < n; g++)
        {
            if (geoms.Types[g] == GeomPlane) continue;
            var (sx, _) = Project(xpos[g * 3 + 0], xpos[g * 3 + 1], 0);
            double height = Math.Max(0.0, xpos[g * 3 + 2]);
            double r = (14 + geoms.Sizes[g * 3] * scale) / (1 + height * 1.6);
            Stage.Children.Add(new Ellipse
            {
                Width = r * 2.2, Height = r * 0.7,
                Fill = new SolidColorBrush(Color.FromArgb((byte)(70 / (1 + height)), 0, 0, 0)),
            }.At(sx - r * 1.1, groundY - r * 0.35));
        }

        // --- Painter's algorithm: far drawn first (ascending depth) ---
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
            Depth(xpos[a * 3], xpos[a * 3 + 1]).CompareTo(Depth(xpos[b * 3], xpos[b * 3 + 1])));

        foreach (int g in order)
        {
            int type = geoms.Types[g];
            if (type == GeomPlane) continue;

            // Depth shading: nearer = brighter, for a 3D feel.
            double depth = Math.Clamp(0.7 + Depth(xpos[g * 3], xpos[g * 3 + 1]) * 0.6, 0.4, 1.0);
            byte lit(byte c) => (byte)Math.Clamp(c * depth, 0, 255);
            var fill = new SolidColorBrush(Color.FromRgb(lit(0xDD), lit(0xE2), lit(0xEC)));
            var edge = new SolidColorBrush(Color.FromRgb(lit(0x6a), lit(0x70), lit(0x86)));

            double px = xpos[g * 3 + 0], py = xpos[g * 3 + 1], pz = xpos[g * 3 + 2];
            // Local axis columns of the row-major 3x3 rotation.
            double zx = xmat[g * 9 + 2], zy = xmat[g * 9 + 5], zz = xmat[g * 9 + 8];

            switch (type)
            {
                case GeomCapsule:
                case GeomCylinder:
                {
                    double R = geoms.Sizes[g * 3 + 0], half = geoms.Sizes[g * 3 + 1];
                    var (ax, ay) = Project(px + zx * half, py + zy * half, pz + zz * half);
                    var (bx, by) = Project(px - zx * half, py - zy * half, pz - zz * half);
                    var capsule = MakeLine(ax, ay, bx, by, default, R * 2 * scale);
                    capsule.Stroke = fill;
                    Stage.Children.Add(capsule);
                    break;
                }
                case GeomBox:
                {
                    double sx = geoms.Sizes[g * 3], sy = geoms.Sizes[g * 3 + 1], sz = geoms.Sizes[g * 3 + 2];
                    double xx = xmat[g * 9 + 0], xy = xmat[g * 9 + 3], xz = xmat[g * 9 + 6];
                    double yx = xmat[g * 9 + 1], yy = xmat[g * 9 + 4], yz = xmat[g * 9 + 7];
                    var pts = new System.Windows.Point[8];
                    int c = 0;
                    for (int dx = -1; dx <= 1; dx += 2)
                    for (int dy = -1; dy <= 1; dy += 2)
                    for (int dz = -1; dz <= 1; dz += 2)
                    {
                        double wx = px + dx * sx * xx + dy * sy * yx + dz * sz * zx;
                        double wy = py + dx * sx * xy + dy * sy * yy + dz * sz * zy;
                        double wz = pz + dx * sx * xz + dy * sy * yz + dz * sz * zz;
                        var (X, Y) = Project(wx, wy, wz);
                        pts[c++] = new System.Windows.Point(X, Y);
                    }
                    var poly = new Polygon { Fill = fill, Stroke = edge, StrokeThickness = 1.5 };
                    foreach (var pt in ConvexHull(pts)) poly.Points.Add(pt);
                    Stage.Children.Add(poly);
                    break;
                }
                default: // sphere / ellipsoid
                {
                    double R = geoms.Sizes[g * 3 + 0] * scale;
                    var (X, Y) = Project(px, py, pz);
                    Stage.Children.Add(new Ellipse { Width = R * 2, Height = R * 2, Fill = fill }
                        .At(X - R, Y - R));
                    break;
                }
            }
        }
    }

    // Andrew's monotone chain — silhouette of the 8 projected box corners.
    private static System.Windows.Point[] ConvexHull(System.Windows.Point[] p)
    {
        var pts = (System.Windows.Point[])p.Clone();
        Array.Sort(pts, (a, b) => a.X != b.X ? a.X.CompareTo(b.X) : a.Y.CompareTo(b.Y));
        var hull = new System.Windows.Point[pts.Length * 2];
        int k = 0;
        double Cross(System.Windows.Point o, System.Windows.Point a, System.Windows.Point b)
            => (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
        for (int i = 0; i < pts.Length; i++)
        {
            while (k >= 2 && Cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) k--;
            hull[k++] = pts[i];
        }
        for (int i = pts.Length - 2, t = k + 1; i >= 0; i--)
        {
            while (k >= t && Cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) k--;
            hull[k++] = pts[i];
        }
        return hull[..(k - 1)];
    }

    private void DrawRobot(double x, double groundY, float skill, bool fallen)
    {
        var body = Color.FromRgb(0xE8, 0xE8, 0xF0);
        var dark = Color.FromRgb(0x55, 0x55, 0x70);

        // Joint wobble shrinks as the robot gets smarter.
        double wobble = (1 - skill) * 0.7;
        double phase = _clock * Math.PI;
        double lean = fallen ? 1.4 : Math.Sin(phase * 0.5) * wobble * 0.3;

        double hipY = groundY - 90 + (fallen ? 55 : Math.Abs(Math.Sin(phase)) * -4);
        double hipX = x;

        // Legs: opposite-phase swings; stride length grows with skill.
        double stride = 18 + 26 * skill;
        for (int side = 0; side < 2; side++)
        {
            double p = phase + side * Math.PI;
            double noise = wobble * (_rng.NextDouble() - 0.5) * 14;
            double kneeX = hipX + Math.Sin(p) * stride * 0.5 + noise;
            double kneeY = hipY + 45;
            double footX = kneeX + Math.Sin(p - 0.6) * stride * 0.6 + noise;
            double footY = fallen ? kneeY + 20 : Math.Min(groundY, kneeY + 45 - Math.Max(0, Math.Sin(p)) * 12);
            var c = side == 0 ? dark : body;
            Stage.Children.Add(MakeLine(hipX, hipY, kneeX, kneeY, c, 9));
            Stage.Children.Add(MakeLine(kneeX, kneeY, footX, footY, c, 9));
            Stage.Children.Add(MakeLine(footX, footY, footX + 16, footY, c, 9));
        }

        // Torso + head, leaning with `lean`.
        double topX = hipX + Math.Sin(lean) * 70, topY = hipY - Math.Cos(lean) * 70;
        Stage.Children.Add(MakeLine(hipX, hipY, topX, topY, body, 14));
        var head = new Ellipse { Width = 34, Height = 34, Fill = new SolidColorBrush(body) };
        Canvas.SetLeft(head, topX - 17); Canvas.SetTop(head, topY - 36);
        Stage.Children.Add(head);
        var eye = new Ellipse { Width = 7, Height = 7, Fill = Brushes.Black };
        Canvas.SetLeft(eye, topX + (fallen ? -3 : 3)); Canvas.SetTop(eye, topY - 24);
        Stage.Children.Add(eye);

        // Arms swing opposite to legs.
        for (int side = 0; side < 2; side++)
        {
            double p = phase + side * Math.PI;
            double shX = topX, shY = topY + 12;
            double handX = shX - Math.Sin(p) * (12 + 18 * skill) + wobble * (_rng.NextDouble() - 0.5) * 10;
            double handY = shY + 38;
            Stage.Children.Add(MakeLine(shX, shY, handX, handY, side == 0 ? dark : body, 7));
        }
    }

    private static Line MakeLine(double x1, double y1, double x2, double y2, Color c, double thick)
        => new()
        {
            X1 = x1, Y1 = y1, X2 = x2, Y2 = y2,
            Stroke = new SolidColorBrush(c),
            StrokeThickness = thick,
            StrokeStartLineCap = PenLineCap.Round,
            StrokeEndLineCap = PenLineCap.Round,
        };
}

internal static class CanvasExtensions
{
    /// <summary>Position a shape on the Canvas and return it (fluent helper).</summary>
    public static T At<T>(this T el, double left, double top) where T : UIElement
    {
        Canvas.SetLeft(el, left);
        Canvas.SetTop(el, top);
        return el;
    }
}
