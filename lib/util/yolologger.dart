import 'dart:developer';

import 'package:pytorch_lite/pytorch_lite.dart';

class YoloLogger {
  static void logDetectionResults(List<ResultObjectDetection> results) {
    log('Total detections: ${results.length}', name: 'YoloDetection');

    for (var i = 0; i < results.length; i++) {
      final result = results[i];
      log('''
Detection #${i + 1}:
  Class Index: ${result.classIndex}
  Class Name: ${result.className ?? 'N/A'}
  Confidence Score: ${(result.score * 100).toStringAsFixed(2)}%
  Rectangle: 
    - x: ${result.rect.left.toStringAsFixed(2)}
    - y: ${result.rect.top.toStringAsFixed(2)}
    - width: ${result.rect.width.toStringAsFixed(2)}
    - height: ${result.rect.height.toStringAsFixed(2)}
''', name: 'YoloDetection');
    }
  }

  static void logSingleDetection(ResultObjectDetection detection,
      {int? index}) {
    final indexStr = index != null ? ' #${index + 1}' : '';
    log('''
Detection$indexStr Details:
  Class Index: ${detection.classIndex}
  Class Name: ${detection.className ?? 'N/A'}
  Confidence Score: ${(detection.score * 100).toStringAsFixed(2)}%
  Rectangle: 
    - x: ${detection.rect.left.toStringAsFixed(2)}
    - y: ${detection.rect.top.toStringAsFixed(2)}
    - width: ${detection.rect.width.toStringAsFixed(2)}
    - height: ${detection.rect.height.toStringAsFixed(2)}
''', name: 'YoloDetection');
  }
}
