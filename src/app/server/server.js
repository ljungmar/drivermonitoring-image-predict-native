var express = require('express');
var app = express();
var cors = require('cors');
var multer = require('multer');
var fs = require('fs');
var path = require('path');

var os = require('os');

var imageSize = '256';
var batchSize = '256';
var bufferSize = '256';
var epochs = '10';
var modelType = 'InceptionNet';
var predictDir = 'PREDICT_DIRECTORY';

var datasetType;
var directoryPath;

app.use(cors());

var storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadFolderPath = './uploads';
    if (!fs.existsSync(uploadFolderPath)) {
      fs.mkdirSync(uploadFolderPath);
    }
    cb(null, uploadFolderPath); // Set the destination folder for uploaded files
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname); // Set the filename with a timestamp
  }
});

var upload = multer({ storage: storage });
var spawn = require('child_process').spawn;

app.use(function(req, res, next) {
  console.log('Incoming request:', req.method, req.url, req.body);
  next();
});

app.post("/upload-file", upload.single('file'), async (req, res) => {

  try {
    console.log("request body: ", __dirname);
    let accuracy = null;
    let loss = null;
    let prediction = null;
    const writeStreams = [];
    // Check if there are uploaded files in req.files array
    const file = req.file;
    await test();
    await train();
    console.log("filepath: ", file.path);
    const imagePath = file.path; // Replace 'uploaded_image.jpg' with the desired file name

    const predictionResult = await makePredictions(imagePath);

    console.log("Prediction result is: ", predictionResult);
    return res.status(200).json({ prediction: predictionResult, file });
} catch (error) {
    console.error(`Error: ${error}`);
    return res.status(500).send("Server error");
  }
});

async function makePredictions(imagePath) {
  let prediction = null;
  return new Promise((resolve, reject) => {
    console.log('Command:', 'python', [
      'experiment.py',
      '-p',
      imagePath, // Replace this with the actual path to the image file
      '-d',
      predictDir,
      '-r',
      imageSize,
      '-c',
      batchSize,
      '-b',
      bufferSize,
      '-e',
      epochs,
      '-m',
      modelType
    ]);
    const pythonProcess = spawn('python', ['experiment.py', '-p', imagePath, '-d',
    predictDir,
    '-r',
    imageSize,
    '-c',
    batchSize,
    '-b',
    bufferSize,
    '-e',
    epochs,
    '-m',
    modelType]);

    let output = "";
    let errorOutput = "";

    pythonProcess.stdout.on('data', (chunk) => {
      output += chunk.toString();
      console.log("Received output chunk: ", chunk.toString());
      // Convert the chunk to a string
      const chunkString = chunk.toString();

      // Check if the chunk contains the prediction information
      if (chunkString.includes('prediction')) {
        const predictionIndex = chunkString.indexOf('prediction');
        const predictionValue = chunkString.substring(predictionIndex + 'prediction'.length).trim();
        prediction = predictionValue.replace(/["{}\n\t\s:]/g, ''); // Remove quotes, curly braces, and whitespace
      }
    });

    pythonProcess.stderr.on('data', (chunk) => {
      errorOutput += chunk.toString();
      console.error("Received error output chunk: ", chunk.toString());
    });

    pythonProcess.on('close', (code) => {
      console.log(`child process exited with code ${code}`);

      // If there was an error, print it and reject the promise
      if (code !== 0) {
        console.error("Python process exited with non-zero code:", code);
        console.error("Error output from Python script:", errorOutput);
        reject(new Error(`An error occurred during prediction. Python exit code: ${code}`));
      }

      // Parse the JSON output received from Python
      let predictionResult = null;
      try {
        predictionResult = JSON.parse(JSON.stringify(output));
      } catch (err) {
        console.error("Error parsing JSON: ", err);
        reject(new Error("An error occurred during parsing the prediction result."));
      }

      console.log("received response: ", prediction);

      resolve(prediction);
    });
  });
}



async function test() {
  datasetType = 'test';

  return new Promise((resolve, reject) => {
    var pythonProcess = spawn('python', ['convert_data.py', '-o', predictDir, '-d', 'statefarm', '-r', imageSize, '-c', batchSize, '-s', datasetType]);

    pythonProcess.stdout.on('data', (data) => {
      console.log('test stdout: ' + data);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error('test stderr: ' + data);
      reject(data);
    });

    pythonProcess.on('close', (code) => {
      console.log('test process exited with code ' + code);
      if (code === 0) {
        resolve();
      } else {
        reject('test process exited with non-zero exit code ' + code);
      }
    });
  });
}

async function train() {
  datasetType = 'train';

  return new Promise((resolve, reject) => {
    var pythonProcess = spawn('python', ['convert_data.py', '-o', predictDir, '-d', 'statefarm', '-r', imageSize, '-c', batchSize, '-s', datasetType]);

    pythonProcess.stdout.on('data', (data) => {
      console.log('train stdout: ' + data);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error('train stderr: ' + data);
      reject(data);
    });

    pythonProcess.on('close', (code) => {
      console.log('train process exited with code ' + code);
      if (code === 0) {
        resolve();
      } else {
        reject('train process exited with non-zero exit code ' + code);
      }
    });
  });
}

app.use(function (err, req, res, next) {
  if (err instanceof multer.MulterError) {
    return res.status(500).json({ result: false, msg: 'File upload error: ' + err.message });
  } else if (err) {
    console.error(err);
    return res.status(500).json({ result: false, msg: 'Server error: ' + err.message });
  }
  next();
});

app.listen(8080, () => {
  // Get the network interfaces of the server
  const networkInterfaces = os.networkInterfaces();

  // Find the IPv4 address
  let ipAddress = '';
  for (const iface in networkInterfaces) {
    for (const details of networkInterfaces[iface]) {
      if (details.family === 'IPv4' && !details.internal) {
        ipAddress = details.address;
        break;
      }
    }
    if (ipAddress) {
      break;
    }
  }

  // Print out the IPv4 address
  console.log(`Server running on port 8080`);
  console.log(`IPv4 Address: ${ipAddress}`);
});
