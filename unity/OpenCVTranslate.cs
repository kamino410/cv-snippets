using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Script to reflect OpenCV extrinsic parameters to the object.
/// * Detail
///     The object attached this script will be translated based on the attitude of the parent object.
/// * Coordinate
///     Assume that Unity coordinate and OpenCV coordinate share X / Z axises (Y axis is inversed).
/// </summary>
[ExecuteInEditMode]
public class OpenCVTranslate : MonoBehaviour {
    public float RotationX;
    public float RotationY;
    public float RotationZ;

    public float TranslationX;
    public float TranslationY;
    public float TranslationZ;

    void Start() {
    }

    void Update() {
        this.transform.localRotation = new Quaternion();
        var rod = new Vector3(RotationX, -RotationY, RotationZ);
        this.transform.Rotate(rod, rod.magnitude * 180 / Mathf.PI, Space.Self);

        var xdir = this.transform.localRotation * Vector3.right;
        var ydir = this.transform.localRotation * Vector3.up;
        var zdir = this.transform.localRotation * Vector3.forward;
        this.transform.localPosition = -xdir * TranslationX + ydir * TranslationY - zdir * TranslationZ;
    }
}
