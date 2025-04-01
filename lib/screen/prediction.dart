import 'dart:developer';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:test_open_cv/util/prediction_process.dart';
import 'package:test_open_cv/util/preprocessing_process.dart';
import 'package:test_open_cv/util/utility.dart';
import 'package:image/image.dart' as img;
import 'package:test_open_cv/util/yolologger.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:loading_animation_widget/loading_animation_widget.dart';

class MyPredictionScreen extends StatefulWidget {
  const MyPredictionScreen({super.key});

  @override
  State<MyPredictionScreen> createState() => _MyPredictionScreenState();
}

class _MyPredictionScreenState extends State<MyPredictionScreen> {
  File? _image;
  // ClassificationModel? classificationModel;
  ModelObjectDetection? objectModel;
  late List<ResultObjectDetection> yoloResults;
  List<Uint8List> images = [];
  List<Uint8List> images_prediction = [];
  bool isLoadingModel = false;
  bool isPredictionProgress = false;
  Uint8List? imageBytes;
  double? processingTime;

  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    initializeModels();
    PredictionProcess.instance.init();
  }

  Future<void> initializeModels() async {
    // classificationModel = await Utility.loadModel(
    //     widget.settingsPredictionModel.modelPredictionResnet);
    objectModel = await Utility.loadYoloModel();
    if (!mounted) return;
    setState(() {
      isLoadingModel = true;
      yoloResults = [];
    });
  }

  Future<void> _pickImage() async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.gallery,
      );

      if (pickedFile != null) {
        setState(() {
          _image = File(pickedFile.path);
          images.clear();
          isPredictionProgress = true;
        });
        yoloDetection(File(pickedFile.path));
      }
    } catch (e) {
      print('Error picking image: $e');
    }
  }

  Future<void> yoloDetection(File imageFile) async {
    try {
      final totalStopwatch = Stopwatch()..start();

      //!compressFile
      final Uint8List? originalImageBytes =
          await Utility.compressFile(imageFile);
      // final Uint8List originalImageBytes = await imageFile.readAsBytes();

      List<ResultObjectDetection> objDetect =
          await objectModel!.getImagePrediction(
        originalImageBytes!,
        minimumScore: 0.5,
        iOUThreshold: 0.5,
        boxesLimit: 80,
      );

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
        images.clear(); // Clear previous images
      });

      //! Log yolo detection
      YoloLogger.logDetectionResults(yoloResults);

      //!Preprocessing
      images_prediction = await PreprocessingProcess.preprocessingMethod(
        yoloResults,
        originalImageBytes,
      );

      setState(() {
        images = images_prediction;
        isPredictionProgress = false;
        processingTime = totalStopwatch.elapsedMilliseconds.toDouble() / 1000.0;
      });
      log('Total processing time !!: ${totalStopwatch.elapsedMilliseconds}ms');
    } catch (e) {
      debugPrint('Error in YOLO detection: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error processing image: ${e.toString()}')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Berry Removing Prediction'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(height: 40),
            Container(
              width: 300,
              height: 500,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
              ),
              child: isPredictionProgress
                  ? Center(
                      child: LoadingAnimationWidget.progressiveDots(
                        color: const Color.fromARGB(255, 131, 64, 193),
                        size: 100,
                      ),
                    )
                  : images.isNotEmpty
                      ? Image.memory(
                          images[0], // Display the first image
                          fit: BoxFit.cover,
                        )
                      : const Icon(
                          Icons.image,
                          size: 100,
                          color: Colors.grey,
                        ),
            ),
            const SizedBox(height: 20),
            if (processingTime != null)
              Text(
                  "Processing Time: ${processingTime?.toStringAsFixed(2)} seconds"),
            // Display all images in a scrollable list
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _pickImage,
              child: const Text('Pick Image'),
            ),
            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }
}
