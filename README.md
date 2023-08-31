# drivermonitoring-image-predict-native

Drivermonitoring application to input a image to predict if it contains a distracted or focused driver for Web, Android and iOS

1. Go into the path "drivermonitoring_predict_native/src/app/server/"
2. Input the statefarm dataset directory in this path
3. Go into the "drivermonitoring_predict_native/src/app/server/server.js" file and change to "python" for Windows and Android or "python3" for Mac and iOS, start the server by calling "node server.js" from the terminal inside of this path
4. Go into "drivermonitoring_image_predict_native/src/app/file-upload/api-service.component.ts" and input your ipv4 address
5. Install all the necessary packages
6. Open up another terminal from the root of the directory and call the command "npm install"
7. Execute the app by calling:
Windows: "ionic serve"
Android: "npx cap run android -l --external"
iOS: "npx cap run ios -l --external"
8. A new window will appear on your device presenting you the application
