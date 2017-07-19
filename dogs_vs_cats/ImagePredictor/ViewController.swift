import UIKit
import CoreML

extension UIImage {

	func resize(to newSize: CGSize) -> UIImage {
		UIGraphicsBeginImageContextWithOptions(CGSize(width: newSize.width, height: newSize.height), true, 1.0)
		self.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
		let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
		UIGraphicsEndImageContext()

		return resizedImage
	}

	func pixelData() -> [UInt8]? {
		let dataSize = size.width * size.height * 4
		var pixelData = [UInt8](repeating: 0, count: Int(dataSize))
		let colorSpace = CGColorSpaceCreateDeviceRGB()
		let context = CGContext(data: &pixelData, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 4 * Int(size.width), space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)

		guard let cgImage = self.cgImage else { return nil }
		context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))

		return pixelData
	}
}

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
	
	func getCVPixelBuffer(from image: UIImage, width:Int, height:Int) -> CVPixelBuffer? {
		let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
		var pixelBuffer : CVPixelBuffer?
		let WIDTH = width
		let HEIGHT = height
		
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
		self.imageCaption.text = "Processing..."
		self.imageNameLabel.text = ""
		self.imagePickerController?.dismiss(animated: true, completion: {
			
			
//			let model = mmtest()
//			let buffer = self.getCVPixelBuffer(from:selectedImage, width:150, height:150);
//			guard let modelResult = try? model.prediction(input1: buffer!) else {
//				self.imageCaption.text = "Error with dogs_vs_cats!"
//				self.imageNameLabel.text = ""
//				return;
//			}
//
//			Swift.print(modelResult)
//
//			var result = "RESULT: \(modelResult.output1[0])"
//
//			self.imageNameLabel.text = result


			let vgg16 = n_vgg16()
			let buffer = self.getCVPixelBuffer(from:selectedImage, width:150, height:150);
			guard let vgg16_result = try? vgg16.prediction(input1: buffer!) else {
				self.imageCaption.text = "Error: Level 1"
				self.imageNameLabel.text = ""
				return;
			}

			let vgg16_extract = n_vgg16_extract()
			let param = vgg16_result.output1




			//Input feature input1 was presented as a batch  (length 4) of sequences (length 512)
			// of vectors of length 4, but the model expects vectors of length 8192.
			guard let mlMultiArray = try? MLMultiArray(shape:[8192], dataType:MLMultiArrayDataType.double) else {
				fatalError("Unexpected runtime error. MLMultiArray")
				return;
			}

			for i in 0 ..< param.count {
				mlMultiArray[i] = param[i]
			}

			guard let vgg16_extract_result = try? vgg16_extract.prediction(input1: mlMultiArray) else {
				self.imageCaption.text = "Error: Level 2"
				self.imageNameLabel.text = ""
				return;
			}

			//Swift.print(vgg16_extract_result)

			self.imageCaption.text = "Success"
			self.imageNameLabel.text = "Result:\(vgg16_extract_result.output1)"



		})
	}
}

