import UIKit
import CoreML

class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {

	@IBOutlet weak var imageCaption: UILabel!
	@IBOutlet weak var imageNameLabel: UILabel!
	var imagePickerController:UIImagePickerController?
	@IBOutlet weak var imageView: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
    }

	@IBAction func pickImage(_ sender: Any) {
		let imagePickerController = UIImagePickerController()
		imagePickerController.sourceType = .photoLibrary
		imagePickerController.delegate = self
		self.imagePickerController = imagePickerController;
		self.present(imagePickerController, animated: true) {
		}
	}

	func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
		self.imagePickerController?.dismiss(animated: true, completion: {

		})
	}

	func getCVPixelBuffer(from image: UIImage) -> CVPixelBuffer? {
		let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
		var pixelBuffer : CVPixelBuffer?

		let WIDTH = 224
		let HEIGHT = 224

//		let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
		let status = CVPixelBufferCreate(kCFAllocatorDefault, WIDTH, HEIGHT, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
		guard (status == kCVReturnSuccess) else {
			return nil
		}
		CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
		let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

		let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
//		let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
		let context = CGContext(data: pixelData, width: WIDTH, height: HEIGHT, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
		context?.translateBy(x: 0, y: image.size.height)
		context?.scaleBy(x: 1.0, y: -1.0)

		UIGraphicsPushContext(context!)
		image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
		UIGraphicsPopContext()
		CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

		return pixelBuffer
	}

	func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {

		let selectedImage = info[UIImagePickerControllerOriginalImage] as! UIImage
		imageView.image = selectedImage
		self.imagePickerController?.dismiss(animated: true, completion: {
		})

		let model = Resnet50()
		let buffer = getCVPixelBuffer(from:selectedImage);

		guard let modelResult = try? model.prediction(image: buffer!) else {
			imageCaption.text = "Error!"
			imageNameLabel.text = ""
			return;
		}
		imageCaption.text = "Predicted Image"
		imageNameLabel.text = modelResult.classLabel
	}
}
