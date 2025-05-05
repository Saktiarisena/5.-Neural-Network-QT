use std::collections::HashMap;
use std::fs::File;

use ndarray::{Array1, Array2};
use csv::ReaderBuilder;
use linfa::dataset::Dataset;
use eframe::{egui, App};
use egui_plot::{Line, Plot, PlotPoints};

struct SimpleNeuralNetwork {
    weights: Array2<f64>,
    bias: Array1<f64>,
    learning_rate: f64,
}

impl SimpleNeuralNetwork {
    fn new(input_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let weights = Array2::from_shape_fn((input_size, output_size), |_| rand::random::<f64>() * 0.1);
        let bias = Array1::zeros(output_size);
        Self { weights, bias, learning_rate }
    }

    fn softmax(&self, x: &Array1<f64>) -> Array1<f64> {
        let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|v| (v - max).exp());
        let sum = exp_x.sum();
        exp_x / sum
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let logits = input.dot(&self.weights) + &self.bias;
        self.softmax(&logits)
    }

    fn train(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
                let output = self.forward(&input.to_owned());
                let error = &output - &target;
                let grad = &output * &(1.0 - &output) * &error;

                let input_col = input.to_owned().insert_axis(ndarray::Axis(1));
                let grad_row = grad.clone().insert_axis(ndarray::Axis(0));
                let grad_matrix = input_col.dot(&grad_row);

                self.weights = &self.weights - &(grad_matrix * self.learning_rate);
                self.bias = &self.bias - &(grad * self.learning_rate);
            }
        }
    }

    fn predict_class(&self, input: &Array1<f64>) -> usize {
        let output = self.forward(input);
        output.iter()
              .enumerate()
              .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
              .map(|(idx, _)| idx)
              .unwrap_or(0)
    }
}

pub struct RiceApp {
    file_path: String,
    learning_rate: f64,
    epochs: usize,
    predictions: Vec<f64>,
    actuals: Vec<f64>,
    mse: f64,
    accuracy: f64, // Add accuracy field
    training_done: bool,
    class_names: Vec<String>,
}

impl Default for RiceApp {
    fn default() -> Self {
        let file_path = "data/Rice_MSC_Dataset_sample.csv".to_string();
        let learning_rate = 0.1;
        let epochs = 1000;
        let (predictions, actuals, mse, accuracy, class_names) = train_and_predict(&file_path, learning_rate, epochs);

        Self {
            file_path,
            learning_rate,
            epochs,
            predictions,
            actuals,
            mse,
            accuracy,
            training_done: true,
            class_names,
        }
    }
}

impl App for RiceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Rice Classifier Trainer");
            ui.horizontal(|ui| {
                ui.label("Learning Rate:");
                ui.add(egui::DragValue::new(&mut self.learning_rate).speed(0.01).clamp_range(0.001..=1.0));
                ui.label("Epochs:");
                ui.add(egui::DragValue::new(&mut self.epochs).speed(10).clamp_range(100..=5000));
            });

            if ui.button("Train Model").clicked() {
                self.train_model();
            }

            if self.training_done && !self.predictions.is_empty() {
                Plot::new("predictions_plot")
                    .view_aspect(2.0)
                    .show(ui, |plot_ui| {
                        let pred_points: PlotPoints = self.predictions.iter().enumerate().map(|(i, &y)| [i as f64, y]).collect();
                        let actual_points: PlotPoints = self.actuals.iter().enumerate().map(|(i, &y)| [i as f64, y]).collect();

                        let pred_line = Line::new(pred_points).name("Predicted");
                        let actual_line = Line::new(actual_points).name("Actual");

                        plot_ui.line(pred_line);
                        plot_ui.line(actual_line);
                    });
                ui.label(format!("Mean Squared Error: {:.4}", self.mse));
                ui.label(format!("Accuracy: {:.2}%", self.accuracy * 100.0));
                
                ui.label("Class Names:");
                for (i, name) in self.class_names.iter().enumerate() {
                    ui.label(format!("Class {}: {}", i, name));
                }
            }
        });
    }
}

impl RiceApp {
    fn train_model(&mut self) {
        self.training_done = false;
        let (predictions, actuals, mse, accuracy, class_names) = train_and_predict(&self.file_path, self.learning_rate, self.epochs);
        self.predictions = predictions;
        self.actuals = actuals;
        self.mse = mse;
        self.accuracy = accuracy;
        self.class_names = class_names;
        self.training_done = true;
    }
}

fn train_and_predict(path: &str, lr: f64, epochs: usize) -> (Vec<f64>, Vec<f64>, f64, f64, Vec<String>) {
    let file = File::open(path).expect("Gagal membuka file");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut records = Vec::new();

    for result in rdr.records() {
        let record = result.unwrap();
        let solidity: f64 = record[0].parse().unwrap();
        let aspect_ratio: f64 = record[1].parse().unwrap();
        let roundness: f64 = record[2].parse().unwrap();
        let compactness: f64 = record[3].parse().unwrap();
        let class = record[4].to_string();
        records.push((solidity, aspect_ratio, roundness, compactness, class));
    }

    let mut features = Array2::<f64>::zeros((records.len(), 4));
    let mut labels = Vec::new();
    let mut class_map = HashMap::new();
    let mut class_names = Vec::new();
    let mut class_counter = 0;

    for (i, (solidity, aspect_ratio, roundness, compactness, class)) in records.iter().enumerate() {
        features[[i, 0]] = *solidity;
        features[[i, 1]] = *aspect_ratio;
        features[[i, 2]] = *roundness;
        features[[i, 3]] = *compactness;

        let class_idx = *class_map.entry(class.clone()).or_insert_with(|| {
            let idx = class_counter;
            class_counter += 1;
            class_names.push(class.clone());
            idx
        });
        labels.push(class_idx);
    }

    let num_classes = class_map.len();
    let one_hot_labels = {
        let mut oh = Array2::<f64>::zeros((labels.len(), num_classes));
        for (i, &label) in labels.iter().enumerate() {
            oh[[i, label]] = 1.0;
        }
        oh
    };

    let dataset = Dataset::new(features.clone(), one_hot_labels);
    let (train, test) = dataset.split_with_ratio(0.8);
    let mut nn = SimpleNeuralNetwork::new(4, num_classes, lr);
    nn.train(train.records(), train.targets(), epochs);

    let predictions: Vec<usize> = test.records().outer_iter().map(|s| nn.predict_class(&s.to_owned())).collect();
    let actuals: Vec<usize> = test.targets().outer_iter().map(|row| {
        row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }).collect();

    // Calculate accuracy
    let correct_predictions = predictions.iter().zip(actuals.iter()).filter(|(p, a)| p == a).count();
    let accuracy = correct_predictions as f64 / predictions.len() as f64;
    let mse = predictions.iter().zip(actuals.iter()).map(|(p, a)| (*p as f64 - *a as f64).powi(2)).sum::<f64>() / predictions.len() as f64;

    // Print classification results to terminal
    println!("\nClassification Results:");
    println!("{:<10} {:<15} {:<15} {:<10}", "Index", "Predicted", "Actual", "Correct");
    for (i, ((pred, actual), correct)) in predictions.iter().zip(actuals.iter()).zip(predictions.iter().zip(actuals.iter()).map(|(p, a)| p == a)).enumerate() {
        println!("{:<10} {:<15} {:<15} {:<10}", 
            i, 
            class_names[*pred], 
            class_names[*actual],
            if correct { "✓" } else { "✗" }
        );
    }

    println!("\nModel Performance:");
    println!("Accuracy: {:.2}%", accuracy * 100.0);
    println!("Correct predictions: {}/{}", correct_predictions, predictions.len());
    println!("Mean Squared Error: {:.4}", mse);
    
    println!("\nClass Mapping:");
    for (i, name) in class_names.iter().enumerate() {
        println!("Class {}: {}", i, name);
    }

    (
        predictions.iter().map(|&x| x as f64).collect(),
        actuals.iter().map(|&x| x as f64).collect(),
        mse,
        accuracy,
        class_names
    )
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Rice Classifier",
        options,
        Box::new(|_cc| Box::new(RiceApp::default())),
    )
}