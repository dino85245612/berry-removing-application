import 'dart:developer';

import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:material_design_icons_flutter/material_design_icons_flutter.dart';
import 'package:test_open_cv/screen/live_camera.dart';
import 'package:test_open_cv/screen/prediction.dart';

class NavigationMenu extends StatefulWidget {
  const NavigationMenu({
    Key? key,
  }) : super(key: key);

  @override
  State<NavigationMenu> createState() => _NavigationMenuState();
}

class _NavigationMenuState extends State<NavigationMenu> {
  GlobalKey btnKey = GlobalKey();
  // Map visible destinations to screens, skipping the hidden ones
  final List<int> navigationIndexMapping = [
    0,
    1,
  ]; // Map destinations to screens

  @override
  Widget build(BuildContext context) {
    final controller = Get.put(NavigationController());

    return Scaffold(
      bottomNavigationBar: Obx(
        () => NavigationBar(
          height: 80,
          elevation: 0,
          selectedIndex: navigationIndexMapping
                  .contains(controller.selectedIndex.value)
              ? navigationIndexMapping.indexOf(controller.selectedIndex.value)
              : 1,
          onDestinationSelected: (index) {
            int mappedIndex = navigationIndexMapping[index];
            controller.setSelectedIndex(mappedIndex);
          },
          destinations: [
            NavigationDestination(icon: Icon(MdiIcons.book), label: "Image"),
            NavigationDestination(
                icon: Icon(MdiIcons.counter, key: btnKey),
                label: "Live Camera"),
          ],
        ),
      ),
      body: Obx(() => controller.screens[controller.selectedIndex.value]),
    );
  }
}

class NavigationController extends GetxController {
  final Rx<int> selectedIndex = 0.obs;
  // final Function(String) setLocale;

  List<Widget> screens = [];

  // NavigationController(this.setLocale) {
  NavigationController() {
    screens = [
      const MyPredictionScreen(), // Index 0
      const MyLiveCameraScreen(), // Index 1
    ];
  }

  void setSelectedIndex(int index) {
    if (index >= 0 && index < screens.length) {
      selectedIndex.value = index;
    } else {
      log("Invalid index: $index");
    }
  }
}
