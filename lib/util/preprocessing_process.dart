import 'dart:developer';
import 'dart:typed_data';

import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:test_open_cv/model/pixel_position.dart';
import 'package:test_open_cv/model/position_data.dart';
import 'package:test_open_cv/util/prediction_process.dart';
import 'package:test_open_cv/util/utility.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:image/image.dart' as img;

class PreprocessingProcess {
  static Future<List<Uint8List>> preprocessingMethod(
    List<ResultObjectDetection> detectionResult,
    Uint8List imageBytes,
  ) async {
    final totalStopwatch = Stopwatch()..start();

    //!get bunch position
    log('Starting preprocessing method...');
    final bunchPositionStopwatch = Stopwatch()..start();
    final bunchPosition = Utility.getBunchPosition(detectionResult);
    log('Bunch position detection took: ${bunchPositionStopwatch.elapsedMilliseconds}ms');

    //!decode to Mat file
    final decodeStopwatch = Stopwatch()..start();
    cv.Mat originalMat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
    log('Image decode took: ${decodeStopwatch.elapsedMilliseconds}ms');

    int imageWidth = originalMat.cols;
    int imageHeight = originalMat.rows;

    //!convert rect to correct pixels
    final bunchPixels = Utility.convertToPixels(
      bunchPosition!,
      imageWidth,
      imageHeight,
    );

    List<Uint8List> images = [];
    List<img.Image> listImages = [];
    List<ResultObjectDetection> berryPositions = [];

    try {
      //! Process each berry separately
      final totalBerryStopwatch = Stopwatch()..start();
      int berryCount = 0;
      for (var detection in detectionResult) {
        if (detection.classIndex == 1) {
          berryCount++;
          final berryStopwatch = Stopwatch()..start();

          //!Create a fresh copy of the original image for each berry
          cv.Mat imageMat = originalMat.clone();
          await processBerry(
            detection,
            imageMat,
            imageWidth,
            imageHeight,
            bunchPixels,
            images,
            listImages,
            berryCount,
          );

          berryPositions.add(detection);
          log('Berry $berryCount total processing took: ${berryStopwatch.elapsedMilliseconds}ms');
        }
      }

      log('$berryCount Total Berry Processing time took: ${totalBerryStopwatch.elapsedMilliseconds}ms');

      final predictStopwatch = Stopwatch()..start();

      //! Predict each berry
      Uint8List byteBerryShouldBeRemoved = await PredictionProcess.instance
          .predict(
              listImages, berryPositions, imageWidth, imageHeight, originalMat);

      log('Predicting berry and total processing took: ${predictStopwatch.elapsedMilliseconds}ms');
      log('Total processing time: ${totalStopwatch.elapsedMilliseconds}ms');

      return [byteBerryShouldBeRemoved];
      // return images;
    } finally {
      log("Cleaning up....");
      originalMat.dispose();
      PredictionProcess.instance.dispose();
    }
  }

  static Future<void> processBerry(
    ResultObjectDetection berry,
    cv.Mat imageMat,
    int imageWidth,
    int imageHeight,
    PixelPosition bunchPixels,
    List<Uint8List> images,
    List<img.Image> listImages,
    int berryIndex,
  ) async {
    final berryProcessStopwatch = Stopwatch()..start();

    try {
      //! Convert berry position to pixels
      final berryPixels = Utility.convertToPixels(
        PositionData(
          x: berry.rect.left,
          y: berry.rect.top,
          width: berry.rect.width,
          height: berry.rect.height,
        ),
        imageWidth,
        imageHeight,
      );
      //! Create points and fill poly for this berry
      final polyStopwatch = Stopwatch()..start();

      final points = Utility.createRectanglePoints(
        berryPixels.left,
        berryPixels.top,
        berryPixels.width,
        berryPixels.height,
      );

      final vecPoints = cv.VecVecPoint.fromList([points]);
      final color = cv.Scalar(255, 255, 255);

      //!Draw white polygon
      cv.fillPoly(
        imageMat,
        vecPoints,
        color,
        lineType: cv.LINE_8,
        shift: 0,
        offset: cv.Point(0, 0),
      );
      log('Berry $berryIndex polygon creation and fill took: ${polyStopwatch.elapsedMilliseconds}ms');

      //!Crop & resize image
      final cropRect = cv.Rect(
          bunchPixels.left.round(),
          bunchPixels.top.round(),
          bunchPixels.width.round(),
          bunchPixels.height.round());

      cv.Mat imageCrop = cv.Mat.fromMat(imageMat, roi: cropRect, copy: true);

      cv.resize(
        imageCrop,
        (224, 224),
        interpolation: cv.INTER_LINEAR,
        dst: imageCrop,
      );

      //! Encode image
      final encodeCropStopwatch = Stopwatch()..start();
      final beforeEncodeCropStopwatch = Stopwatch()..start();

      final bytes = cv.imencode(".png", imageCrop).$2;
      var image = img.decodeImage(bytes);

      log('Berry $berryIndex just endcode eieieieiie: ${beforeEncodeCropStopwatch.elapsedMilliseconds}ms');

      images.add(img.encodeJpg(image!, quality: 85));

      log("preprocess width: ${image.width}");
      log("preprocess height: ${image.height}");
      listImages.add(image);

      log('Berry $berryIndex encode and crop took: ${encodeCropStopwatch.elapsedMilliseconds}ms');
    } finally {
      imageMat.dispose();
    }
    log('Berry $berryIndex total process took: ${berryProcessStopwatch.elapsedMilliseconds}ms');
  }
}
