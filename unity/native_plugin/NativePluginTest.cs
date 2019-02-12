using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class MyCamera : MonoBehaviour
{
    [DllImport("mylib")]
    private static extern int getNumber();

    void Start() {
    }

    void Update() {
    }
}
