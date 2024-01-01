// Get a reference to the button and the div
const runButton = document.getElementById("run-button");
const nrOfHiddenLayers = document.getElementById("hidden-layers");
const nrOfEpochs = document.getElementById("epochs");
const feature = document.getElementById("feature");

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const carsDataResponse = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  const carsData = await carsDataResponse.json();
  const cleaned = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
      Weight_in_lbs: car.Weight_in_lbs,
    }))
    .filter(
      (car) =>
        car.mpg != null && car.horsepower != null && car.Weight_in_lbs != null
    );

  const metric = cleaned.map((car) => ({
    Lper100km: 235.214 / car.mpg,
    horsepower: car.horsepower,
    Weight_in_Kg: car.Weight_in_lbs * 0.45359237,
  }));

  console.log(metric);

  return metric;
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(
    tf.layers.dense({
      inputShape: [1],
      units: 1,
      useBias: true,
      name: "Input_Layer",
    })
  );

  // Add hidden layers
  switch (nrOfHiddenLayers.value) {
    case "1":
      model.add(
        tf.layers.dense({
          units: 50,
          activation: "relu",
          name: "Hidden_Layer_1",
        })
      );
      break;
    case "2":
      model.add(
        tf.layers.dense({
          units: 50,
          activation: "relu",
          name: "Hidden_Layer_1",
        })
      );
      model.add(
        tf.layers.dense({
          units: 50,
          activation: "relu",
          name: "Hidden_Layer_2",
        })
      );
      break;
    case "3":
      model.add(
        tf.layers.dense({
          units: 50,
          activation: "relu",
          name: "Hidden_Layer_1",
        })
      );
      model.add(
        tf.layers.dense({
          units: 50,
          activation: "relu",
          name: "Hidden_Layer_2",
        })
      );
      model.add(
        tf.layers.dense({
          units: 50,
          activation: "relu",
          name: "Hidden_Layer_3",
        })
      );
      break;
  }

  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true, name: "Output_Layer" }));

  return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map((d) => d[feature.value]);
    const labels = data.map((d) => d.Lper100km);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"],
  });

  const batchSize = 32;
  const epochs = nrOfEpochs.value;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const xsNorm = tf.linspace(0, 1, 100);
    const predictions = model.predict(xsNorm.reshape([100, 1]));

    const unNormXs = xsNorm.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = predictions.mul(labelMax.sub(labelMin)).add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d) => ({
    x: d[feature.value],
    y: d.Lper100km,
  }));

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    },
    {
      xLabel: `${feature.value}`,
      yLabel: "Consumption in L/100km",
      height: 300,
    }
  );
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  console.log(feature.value);
  const values = data.map((d) => ({
    x: d[feature.value],
    y: d.Lper100km,
  }));

  tfvis.render.scatterplot(
    { name: `Datapoints` },
    { values },
    {
      xLabel: `${feature.value}`,
      yLabel: "Consumption in L/100km",
      height: 392,
    }
  );

  // More code will be added below
  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log("Done Training");

  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData);
  // Clear the visor
}

// Set up an event listener on the button
runButton.addEventListener("click", run);
