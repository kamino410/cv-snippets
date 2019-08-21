import UIKit
import Vision


let modelURL = Bundle.main.url(forResource: "YOLOv3Tiny", withExtension: "mlmodelc")!

guard let model = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL)) else {
    fatalError()
}

func createRequest(model: VNCoreMLModel) -> VNCoreMLRequest{
    return VNCoreMLRequest(model: model, completionHandler: { (req, err) in
        DispatchQueue.main.async(execute: {
            guard let results = req.results as? [VNRecognizedObjectObservation] else {
                fatalError("Error results")
            }
            for result in results {
                print("\(result.confidence) : ", terminator: "")
                let len = result.labels.count > 5 ? 5 : result.labels.count
                for i in 0..<len{
                    print("\(result.labels[i].identifier), ", terminator: "")
                }
                print()
            }
            print()
        })
    })
}

let img1 = UIImage(named: "sample1.jpg")!
let img2 = UIImage(named: "sample2.jpg")!

let handler1 = VNImageRequestHandler(cgImage: img1.cgImage!, options: [:])
let request1 = createRequest(model: model)
try? handler1.perform([request1])

let handler2 = VNImageRequestHandler(cgImage: img2.cgImage!, options: [:])
let request2 = createRequest(model: model)
try? handler2.perform([request2])
