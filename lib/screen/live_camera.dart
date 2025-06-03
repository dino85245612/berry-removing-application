import 'dart:async';
import 'dart:developer';
import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:test_open_cv/util/prediction_process.dart';
import 'package:test_open_cv/util/preprocessing_process.dart';
import 'package:test_open_cv/util/utility.dart';
import 'package:test_open_cv/util/yolologger.dart';

class MyLiveCameraScreen extends StatefulWidget {
  const MyLiveCameraScreen({super.key});

  @override
  State<MyLiveCameraScreen> createState() => _MyLiveCameraScreenState();
}

class _MyLiveCameraScreenState extends State<MyLiveCameraScreen> {
  bool isLoadingCamera = false;
  bool isRecording = false;
  bool isLoadingModel = false;
  bool _isTakingPicture = false;
  bool isProcessing = false;
  bool isPredictionProgress = false;
  File? imageFromCamera;
  Uint8List? testImg;
  ModelObjectDetection? objectModel;
  List<Uint8List> images = [];
  List<Uint8List> imagesPrediction = [];
  double? processingTime;

  CameraController? _controller;
  Timer? _timer;
  late List<ResultObjectDetection> yoloResults;

  @override
  void initState() {
    super.initState();
    initializeCameraController();
    initializeModels();
    PredictionProcess.instance.init();
  }

  Future<void> initializeModels() async {
    objectModel = await Utility.loadYoloModel();
    setState(() {
      isLoadingModel = true;
      yoloResults = [];
    });
  }

  Future<void> initializeCameraController() async {
    log('initializeCameraController');
    setState(() {
      isLoadingCamera = true;
    });

    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      return;
    }
    try {
      _controller = CameraController(
        cameras.first,
        ResolutionPreset.veryHigh,
      );
    } catch (e) {
      log("[Error] medium resolution not supported. Switching to high resolution.");
      _controller = CameraController(
        cameras.first,
        ResolutionPreset.high,
      );
    }

    log('initializeCameraController waiting until controller is initialized');
    await _controller!.initialize();
    log('initializeCameraController controller is initialized');

    if (!mounted) {
      return;
    }
    setState(() {
      isLoadingCamera = false;
    });
  }

  Future<void> startRecording() async {
    if (!mounted) return;

    setState(() {
      isRecording = true;
    });

    _timer = Timer.periodic(Duration(seconds: 5), (Timer t) async {
      if (_isTakingPicture) return;

      setState(() {
        _isTakingPicture = true;
        isProcessing = true;
      });

      try {
        await _controller!.setFlashMode(FlashMode.off);

        final XFile imageFileTake = await _controller!.takePicture();
        Uint8List imageBytes = await imageFileTake.readAsBytes();

        // Verify image bytes are valid
        if (imageBytes.isEmpty) {
          throw Exception('Invalid image data received from camera');
        }

        if (mounted) {
          setState(() {
            testImg = imageBytes;
          });
        }
        await yoloDetectionLiveCamera(imageBytes);
      } catch (e) {
        log('Error taking picture: $e');
        // Clear any potentially corrupted state
        if (mounted) {
          setState(() {
            testImg = null;
            images.clear();
            yoloResults.clear();
          });
        }
      } finally {
        if (!mounted) return;

        setState(() {
          _isTakingPicture = false;
          isProcessing = false;
        });
      }
    });
  }

  Future<void> stopRecording() async {
    if (!mounted) return;
    _timer?.cancel();
    setState(() {
      isRecording = false;
      yoloResults.clear();
    });
  }

  Future<void> yoloDetectionLiveCamera(Uint8List imageFile) async {
    try {
      final totalStopwatch = Stopwatch()..start();

      //!compressFile
      final Uint8List? originalImageBytes =
          await Utility.compressFileListToList(imageFile);

      if (originalImageBytes == null || originalImageBytes.isEmpty) {
        throw Exception('Failed to compress image');
      }

      final yoloStopwatch = Stopwatch()..start();
      List<ResultObjectDetection> objDetect =
          await objectModel!.getImagePrediction(
        originalImageBytes,
        minimumScore: 0.5,
        iOUThreshold: 0.5,
        boxesLimit: 80,
      );
      log('Yolo detection time: ${yoloStopwatch.elapsedMilliseconds}ms');

      if (!mounted) return;

      if (objDetect.isEmpty) {
        setState(() {
          yoloResults = [];
          images = [originalImageBytes];
        });
        return;
      }

      setState(() {
        yoloResults = objDetect;
        images.clear();
        imagesPrediction.clear();
      });

      //! Log yolo detection
      YoloLogger.logDetectionResults(yoloResults);
      final preProcessingStopwatch = Stopwatch()..start();

      //!Preprocessing
      imagesPrediction = await PreprocessingProcess.preprocessingMethod(
        yoloResults,
        originalImageBytes,
      );
      log('Preprocessing time: ${preProcessingStopwatch.elapsedMilliseconds}ms');

      if (!mounted) return;

      setState(() {
        images = List.from(imagesPrediction);
        isPredictionProgress = false;
        processingTime = totalStopwatch.elapsedMilliseconds.toDouble() / 1000.0;
      });
      log('Total processing time !!: ${totalStopwatch.elapsedMilliseconds}ms');
    } catch (e) {
      debugPrint('Error in YOLO detection: $e');
      if (mounted) {
        setState(() {
          images.clear();
          imagesPrediction.clear();
          yoloResults.clear();
        });

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error processing image: ${e.toString()}')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(
        appBar: AppBar(
            title: Text(
          "Removing Prediction Live Camera",
          style: const TextStyle(
            fontWeight: FontWeight.w700,
          ),
        )),
        body: isLoadingCamera
            ? const Center(child: CircularProgressIndicator())
            : Column(
                children: [
                  Expanded(
                    child: LayoutBuilder(builder: (context, constraints) {
                      return GestureDetector(
                        child: Stack(
                          children: [
                            // Full-screen camera preview
                            SizedBox(
                                width: double.infinity,
                                height: double.infinity,
                                child: CameraPreview(_controller!)),
                            if (testImg != null)
                              Align(
                                alignment: Alignment.topRight,
                                child: Padding(
                                  padding: EdgeInsets.only(
                                    top: screenHeight * 0.02,
                                    right: screenWidth * 0.06,
                                  ),
                                  child: Image.memory(
                                    testImg!,
                                    width: 150, // optional size
                                    height: 200,
                                    fit: BoxFit.cover,
                                  ),
                                ),
                              ),
                            if (images.isNotEmpty)
                              Align(
                                alignment: Alignment.topLeft,
                                child: Padding(
                                  padding: EdgeInsets.only(
                                    top: screenHeight * 0.02,
                                    right: screenWidth * 0.1,
                                  ),
                                  child: Image.memory(
                                    images[0],
                                    width: 150, // optional size
                                    height: 200,
                                    fit: BoxFit.cover,
                                  ),
                                ),
                              ),

                            Positioned(
                              left: screenWidth *
                                  0.43, // Align to the bottom left
                              bottom: screenHeight *
                                  0.06, // Align to the bottom left
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  // Start recording button
                                  if (!isRecording)
                                    FloatingActionButton(
                                      heroTag: 'startRecording',
                                      onPressed: startRecording,
                                      backgroundColor: Colors.blueAccent,
                                      foregroundColor: Colors.white,
                                      child: const Icon(Icons.radio_button_on),
                                    ),
                                  // SizedBox(width: 20),
                                  // Stop recording button
                                  if (isRecording)
                                    FloatingActionButton(
                                      heroTag: 'stopRecording',
                                      onPressed: stopRecording,
                                      backgroundColor: Colors.red,
                                      foregroundColor: Colors.white,
                                      child: const Icon(Icons.stop),
                                    ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      );
                    }),
                  ),
                ],
              ));
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.dispose();
    images.clear();
    imagesPrediction.clear();
    yoloResults.clear();
    super.dispose();
  }
}
