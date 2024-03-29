var express = require('express');
var app = express();
var cors = require('cors');
var multer = require('multer');
var fs = require('fs');
var path = require('path');

var os = require('os');

var imageSize = '64';
var batchSize = '64';
var bufferSize = '64';
var epochs = '10';
var modelType = 'cnn';
var predictDir = 'PREDICT_DIRECTORY';
var python = 'python'; // for Windows: python for Mac: python3

var datasetType;

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
    cb(null, file.originalname); // Set the filename with a timestamp
  }
});

var upload = multer({ storage: storage });
var spawn = require('child_process').spawn;

app.post("/upload-file", upload.single('file'), async (req, res) => {

  try {
    console.log("request body: ", __dirname);
    // Check if there are uploaded files in req.files array
    const file = req.file;
    console.log(req.file.size);
    console.log("filepath: ", file.path);
    const imagePath = file.path;
    const predictionResult = await singularPrediction(imagePath);

    console.log("Prediction result is: " + predictionResult);
    return res.status(200).json({ prediction: predictionResult, file });
} catch (error) {
    console.error(`Error: ${error}`);
    return res.status(500).send("Server error");
  }
});

async function singularPrediction(imagePath) {
  return new Promise((resolve, reject) => {
    var pythonProcess = spawn(python, ['./modelPredictor.py', '-p', imagePath]);
    let prediction = "";
    
    pythonProcess.stdout.on('data', (data) => {
      prediction = String.fromCharCode(...data).replace(/["{}\n\t\s:]/g, '');
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        console.log("Driver is " + prediction);
        resolve(prediction);
      } else {
        reject('predict process exited with non-zero exit code ' + code);
      }
    });
  });
}

async function ifModelExists() {
  let model = "./saved_models/trained_model_64x64.h5";
  let before = new Date().getTime();
  
  if (!fs.existsSync(model)) {
    console.log("No Model is pre-existing, creating a new model.");
    await test();
    await train();
    await makePredictions('./uploads/in-car-focused-001.jpg');
    let after = new Date().getTime();
    let minutes = ((after - before) / 1000) / 60;
    console.log("Initial test, training, and validation completed in: " + minutes.toString() + " minutes.");
  }
  console.log("Model already exists, using the premade model.");
};

async function makePredictions(imagePath) {
  let prediction = null;
  return new Promise((resolve, reject) => {
    console.log('Command:', python, [
      'experiment.py',
      '-p',
      imagePath,
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
    const pythonProcess = spawn(python, [
    'experiment.py', 
    '-p', 
    imagePath, 
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

    let output = "";
    let errorOutput = "";

    pythonProcess.stdout.on('data', (chunk) => {
      output += chunk.toString();
      console.log("Received output chunk: ", chunk.toString());
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
    var pythonProcess = spawn(python, ['convert_data.py', '-o', predictDir, '-d', 'statefarm', '-r', imageSize, '-c', batchSize, '-s', datasetType]);

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
    var pythonProcess = spawn(python, ['convert_data.py', '-o', predictDir, '-d', 'statefarm', '-r', imageSize, '-c', batchSize, '-s', datasetType]);

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

app.listen(8080, async () => {
  try {
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

  await ifModelExists();
  }
  catch (error) {
    console.log("Crash.");
  }
});
