import UIKit
import Vision


let modelURL = Bundle.main.url(forResource: "mnist", withExtension: "mlmodelc")!
print(modelURL)

guard let model = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL)) else {
    fatalError()
}

extension Array where Element : Comparable {
    public func argmax() -> Int {
        return self.enumerated().max{a, b in a.1 < b.1}!.0
    }
}

extension MLMultiArray {
    public func doubleArray() -> [Double] {
        if self.count == 0{
            return [Double]()
        }
        let ptr = self.dataPointer.bindMemory(to: Double.self, capacity: self.count)
        return Array(UnsafeBufferPointer(start: ptr, count: self.count))
    }
}

func createRequest(model: VNCoreMLModel) -> VNCoreMLRequest{
    return VNCoreMLRequest(model: model, completionHandler: { (req, err) in
        DispatchQueue.main.async(execute: {
            guard let results = req.results as? [VNCoreMLFeatureValueObservation] else {
                fatalError("Error results")
            }
            print(results[0].featureValue.multiArrayValue!.doubleArray().argmax())
        })
    })
}

let img = UIImage(named: "0.png")!

let handler = VNImageRequestHandler(cgImage: img.cgImage!, options: [:])
let request = createRequest(model: model)
try? handler.perform([request])
