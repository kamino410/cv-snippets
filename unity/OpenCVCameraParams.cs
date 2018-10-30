using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Script to reflect OpenCV intrinsic parameters to the camera.
/// * Coordinate
///     Assume that Unity coordinate and OpenCV coordinate share X / Z axises (Y axis is inversed).
/// </summary>
[ExecuteInEditMode]
public class OpenCVCameraParams : MonoBehaviour {
    private Camera cam;

    public int ImageWidth;
    public int ImageHeight;

    public float Fx;
    public float Fy;
    public float Cx;
    public float Cy;
    public float Near;
    public float Far;

    void Start() {
        cam = this.GetComponent<Camera>();
    }

    void Update() {
        cam.projectionMatrix = PerspectiveMat();

        float screenAspect = (float)Screen.height / Screen.width;
        float aspect = (float)ImageHeight / ImageWidth;
        if (screenAspect < aspect) {
            float scale = (float)ImageHeight / Screen.height;
            float camWidth = ImageWidth / (Screen.width * scale);
            cam.rect = new Rect((1 - camWidth) / 2, 0, camWidth, 1);
        } else {
            float scale = (float)ImageWidth / Screen.width;
            float camHeight = (float)ImageHeight / (Screen.height * scale);
            cam.rect = new Rect(0, (1 - camHeight) / 2, 1, camHeight);
        }
    }

    Matrix4x4 PerspectiveMat() {
        var m = new Matrix4x4();
        m[0, 0] = 2 * Fx / ImageWidth;
        m[0, 1] = 0;
        m[0, 2] = 1 - 2 * Cx / ImageWidth;
        m[0, 3] = 0;

        m[1, 0] = 0;
        m[1, 1] = 2 * Fy / ImageHeight;
        m[1, 2] = -1 + 2 * Cy / ImageHeight;
        m[1, 3] = 0;

        m[2, 0] = 0;
        m[2, 1] = 0;
        m[2, 2] = -(Far + Near) / (Far - Near);
        m[2, 3] = -2 * Far * Near / (Far - Near);

        m[3, 0] = 0;
        m[3, 1] = 0;
        m[3, 2] = -1;
        m[3, 3] = 0;
        return m;
    }
}
