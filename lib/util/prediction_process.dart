import 'dart:developer';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:test_open_cv/model/position_data.dart';
import 'package:test_open_cv/util/utility.dart';
import 'package:opencv_dart/opencv.dart' as cv;

class PredictionProcess {
  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;

  bool _isInitialized = false;

  PredictionProcess._();
  static final instance = PredictionProcess._();
  factory PredictionProcess() => instance;

  Future<void> init() async {
    if (!_isInitialized) {
      await loadModelPrediction();
    }
  }

  Future<void> dispose() async {
    if (_isInitialized) {
      _sessionOptions?.release();
      _sessionOptions = null;
      _session?.release();
      _session = null;
      OrtEnv.instance.release();

      _isInitialized = false;
    }
  }

  Future<void> loadModelPrediction() async {
    OrtSessionOptions? sessionOptions;

    log("Loading Prediction Model...");
    try {
      OrtEnv.instance.init();
      sessionOptions = OrtSessionOptions();
      final rawAssetFile =
          await rootBundle.load("assets/model/prediction_model.onnx");
      final bytes = rawAssetFile.buffer.asUint8List();
      _session = OrtSession.fromBuffer(bytes, sessionOptions);

      _isInitialized = true;
      log("Load Prediction Model Successful!");
    } catch (e) {
      log("Error loading Color model: $e");
    }
  }

  Float32List imageToFloat32List(
    Image image,
    List<double> mean,
    List<double> std,
  ) {
    var bytes = Float32List(1 * image.height * image.width * 3);
    var buffer = Float32List.view(bytes.buffer);

    int offsetG = image.height * image.width;
    int offsetB = 2 * image.height * image.width;
    int i = 0;
    for (var y = 0; y < image.height; y++) {
      for (var x = 0; x < image.width; x++) {
        Pixel pixel = image.getPixel(x, y);
        buffer[i] = ((pixel.r / 255) - mean[0]) / std[0];
        buffer[offsetG + i] = ((pixel.g / 255) - mean[1]) / std[1];
        buffer[offsetB + i] = ((pixel.b / 255) - mean[2]) / std[2];
        i++;
      }
    }
    return buffer;
  }

  Future<Uint8List> predict(
    List<Image> images,
    List<ResultObjectDetection> berryPositions,
    int imageWidth,
    int imageHeight,
    cv.Mat originalMat,
  ) async {
    const imageSize = 224;
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    log("Start prediction...");
    final inputDataList = images.map((berryImage) {
      final data = imageToFloat32List(berryImage, mean, std);
      return data;
    }).toList();

    final inputShape = [images.length, 3, imageSize, imageSize];
    // log("Input DataList: ${inputDataList[0]}");
    final inputOrt =
        OrtValueTensor.createTensorWithDataList(inputDataList, inputShape);

    final inputs = {'input': inputOrt};
    // final stopwatch = Stopwatch()..start();
    List<OrtValue?>? outputs;

    try {
      log("Running model...");
      final runOptions = OrtRunOptions();
      outputs = await _session?.runAsync(runOptions, inputs);
      inputOrt.release();
      runOptions.release();
    } catch (e) {
      log("Error while running color inference.");
    }

    final predictions = outputs?[0]?.value as List;

    log("Prediction result:\n ${predictions.toString()}");

    //!get index of maximum value
    int indexMaximumValue = sortPredictionResults(predictions);

    if (indexMaximumValue >= 0 && indexMaximumValue < images.length) {
      final drawRectangleStopwatch = Stopwatch()..start();
      log('Returning image at index: $indexMaximumValue');

      //!get position of berry that should be removed
      final selectedBerry = berryPositions[indexMaximumValue];
      final berryRemovePixels = Utility.convertToPixels(
        PositionData(
          x: selectedBerry.rect.left,
          y: selectedBerry.rect.top,
          width: selectedBerry.rect.width,
          height: selectedBerry.rect.height,
        ),
        imageWidth,
        imageHeight,
      );

      //!draw rectangle on image
      cv.Mat imageMatRemoved = originalMat.clone();
      final rectRemovedPosition = cv.Rect(
          berryRemovePixels.left,
          berryRemovePixels.top,
          berryRemovePixels.width,
          berryRemovePixels.height);

      cv.Mat removedImage = cv.rectangle(
          imageMatRemoved, rectRemovedPosition, cv.Scalar(255, 0, 255),
          thickness: 4);

      final bytes = cv.imencode(".png", removedImage).$2;

      log('Draw berry that should be removed and total processing took: ${drawRectangleStopwatch.elapsedMilliseconds}ms');

      return bytes;
    } else {
      log('Index out of bounds: $indexMaximumValue. Total images: ${images.length}');
      return Uint8List(0);
    }
  }

  int sortPredictionResults(List<dynamic> predictions) {
    final indexedValues = predictions
        .asMap()
        .entries
        .map((entry) =>
            MapEntry(entry.key, (entry.value as List<dynamic>)[1] as double))
        .toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    final maxIndex = indexedValues.first.key;
    final maxValue = indexedValues.first.value;
    log("Maximum value found at index $maxIndex with value ${maxValue.toStringAsFixed(4)}");

    return maxIndex;
  }
}
