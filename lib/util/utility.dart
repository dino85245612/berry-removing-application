import 'dart:developer';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:image/image.dart' as img;
import 'package:test_open_cv/model/pixel_position.dart';
import 'package:test_open_cv/model/position_data.dart';
import 'package:test_open_cv/util/prediction_process.dart';

class Utility {
  static Future<Uint8List?> compressFile(File file) async {
    var result = await FlutterImageCompress.compressWithFile(
      file.absolute.path,
      minWidth: 640,
      minHeight: 640,
      quality: 85,
      // rotate: 90,
    );
    return result;
  }

  static Future<ModelObjectDetection?> loadYoloModel() async {
    log("Loading Yolo Model...");
    ModelObjectDetection? objectModel;
    try {
      objectModel = await PytorchLite.loadObjectDetectionModel(
          "assets/model/yolov8.torchscript", 2, 640, 640,
          labelPath: "assets/model/labels.txt",
          objectDetectionModelType: ObjectDetectionModelType.yolov8);
      log("Load Yolo Model Successful!");
      return objectModel;
    } on PlatformException {
      log("only supported for android and ios so far");
    } on Exception {
      log("Error loading yolo model");
    }
    return null;
  }

  static List<cv.Point> createRectanglePoints(
      int startX, int startY, int width, int height) {
    return [
      cv.Point(startX, startY),
      cv.Point(startX + width, startY),
      cv.Point(startX + width, startY + height),
      cv.Point(startX, startY + height),
    ];
  }

  static PositionData? getBunchPosition(
      List<ResultObjectDetection> detections) {
    for (var result in detections) {
      if (result.classIndex == 0) {
        return PositionData(
          x: result.rect.left,
          y: result.rect.top,
          width: result.rect.width,
          height: result.rect.height,
        );
      }
    }
    return null;
  }

  static PixelPosition convertToPixels(
    PositionData position,
    int imageWidth,
    int imageHeight,
  ) {
    return PixelPosition(
      left: (position.x * imageWidth).round(),
      top: (position.y * imageHeight).round(),
      width: (position.width * imageWidth).round(),
      height: (position.height * imageHeight).round(),
    );
  }
}
