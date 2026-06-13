// MainWindow.xaml.cs — three buttons, one robot, zero menus.
// The cartoon robot's walk quality is driven by the backend's skill value:
// low skill = wobbling and falling over, high skill = confident strides.
using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
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

    public MainWindow()
    {
        InitializeComponent();

        _backend = CreateBackend();
        if (_backend.IsReal)
        {
            Title = "Robot School — real MuJoCo physics";
        }
        else
        {
            Title = "Robot School — demo mode";
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
        DrawScene(status, dt);
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

    // ----------------------------------------------------------- Cartoon robot

    private void DrawScene(RobotStatus status, double dt)
    {
        Stage.Children.Clear();
        double w = Stage.ActualWidth, h = Stage.ActualHeight;
        if (w < 50 || h < 50) return;

        // Real engine available → draw the actual MuJoCo rigid-body pose.
        if (_backend.TryGetPose(out float[] xyz, out int[] parents))
        {
            DrawRealRobot(xyz, parents, w, h);
            return;
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

    // Side-on orthographic view of agent 0's real physics state: world +x (walk
    // direction) maps to screen x with the camera tracking the torso, world +z
    // (up) maps to screen y. Bones connect each body to its parent.
    private void DrawRealRobot(float[] xyz, int[] parents, double w, double h)
    {
        const double scale = 200;                 // pixels per meter
        double groundY = h * 0.82;
        int nbody = parents.Length;
        if (nbody < 2 || xyz.Length < nbody * 3) return;

        double rootX = xyz[1 * 3 + 0];            // body 1 = torso
        double camX = w * 0.45;

        // Ground stripes scroll with the robot's real forward progress.
        double scroll = ((rootX * scale) % 60 + 60) % 60;
        for (double x = -scroll; x < w; x += 60)
            Stage.Children.Add(MakeLine(x, groundY + 10, x + 30, groundY + 10,
                                        Color.FromRgb(0x4e, 0x83, 0x3f), 4));
        Stage.Children.Add(MakeLine(0, groundY, w, groundY,
                                    Color.FromRgb(0x44, 0x70, 0x36), 2));

        var bone = Color.FromRgb(0xE8, 0xE8, 0xF0);
        var joint = Color.FromRgb(0x55, 0x55, 0x70);

        double Sx(int b) => camX + (xyz[b * 3 + 0] - rootX) * scale;
        double Sy(int b) => groundY - xyz[b * 3 + 2] * scale;

        for (int b = 1; b < nbody; b++)
        {
            int p = parents[b];
            if (p > 0)
                Stage.Children.Add(MakeLine(Sx(b), Sy(b), Sx(p), Sy(p), bone, 8));
        }
        for (int b = 1; b < nbody; b++)
        {
            // Torso (body 1) gets a head-sized marker so the robot reads as a person.
            double r = b == 1 ? 13 : 5;
            var dot = new Ellipse { Width = r * 2, Height = r * 2, Fill = new SolidColorBrush(joint) };
            Canvas.SetLeft(dot, Sx(b) - r);
            Canvas.SetTop(dot, Sy(b) - r - (b == 1 ? 30 : 0));
            Stage.Children.Add(dot);
        }
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
