// Scene3D.cs — real 3D viewport for the humanoid crowd (WPF Media3D).
//
// Renders the first K agents of the (collision-independent) batch as a row of
// differently-colored humanoids built from the model's actual collision geoms
// (capsule = cylinder + 2 end spheres, box, sphere), each transformed every
// frame by the live geom_xpos / geom_xmat read from physics. Perspective
// camera + lighting + mouse orbit make it genuinely 3D, not a flat projection.
using System;
using System.Collections.Generic;
using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace Aistank.Editor;

public sealed class Scene3D
{
    private const int GeomPlane = 0, GeomSphere = 2, GeomCapsule = 3, GeomCylinder = 5, GeomBox = 6;

    private readonly Model3DGroup _root;
    private readonly GeomModel _geoms;
    private readonly int _agents;
    private readonly float[] _xpos, _xmat;            // scratch for one agent's pose
    private readonly double _spacing;

    // Per (agent, geom) the transforms we update each frame. A capsule owns 3.
    private readonly List<(int agent, int geom, int kind, MatrixTransform3D xf)> _parts = new();
    private static readonly MeshGeometry3D Sphere = BuildSphere(12, 12);
    private static readonly MeshGeometry3D Cylinder = BuildCylinderZ(20);
    private static readonly MeshGeometry3D Box = BuildBox();

    public Scene3D(Model3DGroup root, GeomModel geoms, int geomBufLen, int agents, double spacing)
    {
        _root = root;
        _geoms = geoms;
        _agents = agents;
        _spacing = spacing;
        _xpos = new float[geomBufLen * 3];
        _xmat = new float[geomBufLen * 9];
        Build();
    }

    private void Build()
    {
        _root.Children.Clear();
        _parts.Clear();

        // Lights.
        _root.Children.Add(new AmbientLight(Color.FromRgb(80, 80, 90)));
        _root.Children.Add(new DirectionalLight(Color.FromRgb(230, 230, 220),
            new Vector3D(-0.4, 0.6, -1.0)));

        // Ground plane (big flat box) the crowd shares.
        var groundMat = new DiffuseMaterial(new SolidColorBrush(Color.FromRgb(94, 156, 76)));
        var ground = new GeometryModel3D(Box, groundMat) { BackMaterial = groundMat };
        double half = _agents * _spacing;
        ground.Transform = new MatrixTransform3D(
            Compose(ScaleMat(half, half, 0.02), Identity(), 0, 0, -0.02));
        _root.Children.Add(ground);

        int cols = (int)Math.Ceiling(Math.Sqrt(_agents));
        for (int a = 0; a < _agents; a++)
        {
            Color col = HsvColor(a / (double)Math.Max(_agents, 1));
            var mat = new MaterialGroup();
            mat.Children.Add(new DiffuseMaterial(new SolidColorBrush(col)));
            mat.Children.Add(new SpecularMaterial(new SolidColorBrush(Color.FromRgb(60, 60, 60)), 20));

            for (int g = 0; g < _geoms.Count; g++)
            {
                int type = _geoms.Types[g];
                if (type == GeomPlane) continue;
                if (type == GeomCapsule || type == GeomCylinder)
                {
                    AddPart(a, g, 0, Cylinder, mat);     // body
                    AddPart(a, g, 1, Sphere, mat);       // +z cap
                    AddPart(a, g, 2, Sphere, mat);       // -z cap
                }
                else if (type == GeomBox) AddPart(a, g, 3, Box, mat);
                else AddPart(a, g, 4, Sphere, mat);      // sphere / fallback
            }
        }
    }

    private void AddPart(int a, int g, int kind, MeshGeometry3D mesh, Material mat)
    {
        var xf = new MatrixTransform3D(Matrix3D.Identity);
        // BackMaterial too: makes each solid render regardless of triangle winding
        // (no hollow / inverted-looking shapes).
        _root.Children.Add(new GeometryModel3D(mesh, mat) { BackMaterial = mat, Transform = xf });
        _parts.Add((a, g, kind, xf));
    }

    /// <summary>Pull each agent's geom pose and update every part's transform.</summary>
    public void Update(IRobotBackend backend)
    {
        int cols = (int)Math.Ceiling(Math.Sqrt(_agents));
        int idx = 0;
        // Walk parts grouped by agent: fetch each agent's pose once into reusable
        // scratch buffers (no per-frame allocation).
        for (int a = 0; a < _agents; a++)
        {
            backend.TryGetAgentGeomPose(a, _xpos, _xmat);
            double ox = (a % cols - cols / 2.0) * _spacing;
            double oy = (a / cols - cols / 2.0) * _spacing;

            while (idx < _parts.Count && _parts[idx].agent == a)
            {
                var (_, g, kind, xf) = _parts[idx++];
                // Geom world rotation (MuJoCo row-major R) → WPF row-vector matrix = transpose.
                var R = new Matrix3D(
                    _xmat[g * 9 + 0], _xmat[g * 9 + 3], _xmat[g * 9 + 6], 0,
                    _xmat[g * 9 + 1], _xmat[g * 9 + 4], _xmat[g * 9 + 7], 0,
                    _xmat[g * 9 + 2], _xmat[g * 9 + 5], _xmat[g * 9 + 8], 0,
                    0, 0, 0, 1);
                double px = _xpos[g * 3 + 0] + ox, py = _xpos[g * 3 + 1] + oy, pz = _xpos[g * 3 + 2];
                double s0 = _geoms.Sizes[g * 3 + 0], s1 = _geoms.Sizes[g * 3 + 1], s2 = _geoms.Sizes[g * 3 + 2];

                Matrix3D local;
                switch (kind)
                {
                    case 0: local = ScaleMat(s0, s0, s1); break;                 // capsule body
                    case 1: local = Mul(ScaleMat(s0, s0, s0), TransMat(0, 0, s1)); break;  // +z cap
                    case 2: local = Mul(ScaleMat(s0, s0, s0), TransMat(0, 0, -s1)); break; // -z cap
                    case 3: local = ScaleMat(s0, s1, s2); break;                 // box
                    default: local = ScaleMat(s0, s0, s0); break;                // sphere
                }
                xf.Matrix = Compose(local, R, px, py, pz);
            }
        }
    }

    // ---- small matrix helpers (WPF row-vector convention: p' = p * M) ----
    private static Matrix3D Identity() => Matrix3D.Identity;
    private static Matrix3D ScaleMat(double x, double y, double z)
        => new(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
    private static Matrix3D TransMat(double x, double y, double z)
        => new(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1);
    private static Matrix3D Mul(Matrix3D a, Matrix3D b) { a.Append(b); return a; }
    private static Matrix3D Compose(Matrix3D local, Matrix3D rot, double tx, double ty, double tz)
    {
        var m = local; m.Append(rot);
        m.OffsetX += tx; m.OffsetY += ty; m.OffsetZ += tz;
        return m;
    }

    private static Color HsvColor(double h)
    {
        double r = 0, g = 0, b = 0; double s = 0.65, v = 0.95;
        int i = (int)(h * 6) % 6; double f = h * 6 - Math.Floor(h * 6);
        double p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
        switch (i)
        {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break;
        }
        return Color.FromRgb((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
    }

    // ---- procedural unit meshes ----
    private static MeshGeometry3D BuildSphere(int stacks, int slices)
    {
        var m = new MeshGeometry3D();
        for (int i = 0; i <= stacks; i++)
        {
            double phi = Math.PI * i / stacks;
            for (int j = 0; j <= slices; j++)
            {
                double theta = 2 * Math.PI * j / slices;
                double x = Math.Sin(phi) * Math.Cos(theta);
                double y = Math.Sin(phi) * Math.Sin(theta);
                double z = Math.Cos(phi);
                m.Positions.Add(new Point3D(x, y, z));
                m.Normals.Add(new Vector3D(x, y, z));      // outward (unit sphere)
            }
        }
        int row = slices + 1;
        for (int i = 0; i < stacks; i++)
            for (int j = 0; j < slices; j++)
            {
                int a = i * row + j, b = a + row;
                m.TriangleIndices.Add(a); m.TriangleIndices.Add(b); m.TriangleIndices.Add(a + 1);
                m.TriangleIndices.Add(a + 1); m.TriangleIndices.Add(b); m.TriangleIndices.Add(b + 1);
            }
        return m;
    }

    private static MeshGeometry3D BuildCylinderZ(int slices)
    {
        var m = new MeshGeometry3D();
        for (int j = 0; j <= slices; j++)
        {
            double t = 2 * Math.PI * j / slices, c = Math.Cos(t), s = Math.Sin(t);
            m.Positions.Add(new Point3D(c, s, -1)); m.Normals.Add(new Vector3D(c, s, 0));
            m.Positions.Add(new Point3D(c, s, 1));  m.Normals.Add(new Vector3D(c, s, 0));
        }
        for (int j = 0; j < slices; j++)
        {
            int a = j * 2;
            m.TriangleIndices.Add(a); m.TriangleIndices.Add(a + 1); m.TriangleIndices.Add(a + 2);
            m.TriangleIndices.Add(a + 2); m.TriangleIndices.Add(a + 1); m.TriangleIndices.Add(a + 3);
        }
        return m;
    }

    private static MeshGeometry3D BuildBox()
    {
        var m = new MeshGeometry3D();
        double[,] v = { {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1} };
        for (int i = 0; i < 8; i++) m.Positions.Add(new Point3D(v[i, 0], v[i, 1], v[i, 2]));
        int[] f = { 0,3,2, 0,2,1, 4,5,6, 4,6,7, 0,1,5, 0,5,4,
                    1,2,6, 1,6,5, 2,3,7, 2,7,6, 3,0,4, 3,4,7 };
        foreach (int i in f) m.TriangleIndices.Add(i);
        return m;
    }
}
