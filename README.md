# drivermonitoring-image-predict-native

Drivermonitoring application to input a image to predict if it contains a distracted or focused driver for Web, Android and iOS

Prerequisites: for Android devices: Android Studio, for iOS devices: XCode

1. Go into the path "drivermonitoring_predict_native/src/app/server/"
2. Input the statefarm dataset directory in this path
3. Go into the "drivermonitoring_predict_native/src/app/server/server.js" file and change to "python" for Windows and Android or "python3" for Mac and iOS
4. Start the server by calling "node server.js" from the terminal inside of this path
5. Go into "drivermonitoring_image_predict_native/src/app/file-upload/api-service.component.ts" and input your ipv4 address
6. Install all the necessary packages
7. Open up another terminal from the root of the directory and call the command "npm install"
8. Call the command "ionic build"
9. If your on a iOS device call the command "ionic cap add ios" or if your using on an Android device instead call "ionic cap add android"
10. Execute the app by calling: for Web: "ionic serve", for Android: "npx cap run android -l --external", for iOS: "npx cap run ios -l --external"
11. A new window will appear on your device presenting you with the application
