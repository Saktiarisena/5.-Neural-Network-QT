pub fn train_and_predict_with_manual_data(
    path: &str,
    manual_features: &[[f64; 4]],
    manual_labels: &[String],
    lr: f64,
    epochs: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
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

    // Tambahkan manual data ke records
    for (feature, class) in manual_features.iter().zip(manual_labels.iter()) {
        records.push((feature[0], feature[1], feature[2], feature[3], class.clone()));
    }

    let mut features = Array2::<f64>::zeros((records.len(), 4));
    let mut labels = Vec::new();
    let mut class_map = HashMap::new();
    let mut class_counter = 0;

    for (i, (solidity, aspect_ratio, roundness, compactness, class)) in records.iter().enumerate() {
        features[[i, 0]] = *solidity;
        features[[i, 1]] = *aspect_ratio;
        features[[i, 2]] = *roundness;
        features[[i, 3]] = *compactness;

        let class_idx = *class_map.entry(class.clone()).or_insert_with(|| {
            let idx = class_counter;
            class_counter += 1;
            idx
        });
        labels.push(class_idx);
    }

    let labels = Array1::from(labels).mapv(|x| x as f64);
    let max_label = labels.iter().cloned().fold(f64::MIN, |a, b| a.max(b));

    let (train, test) = Dataset::new(features.clone(), labels.clone()).split_with_ratio(0.8);
    let mut nn = Self::new(4, 1, lr);
    nn.train(train.records(), train.targets(), epochs);

    let predictions: Vec<f64> = test.records().outer_iter().map(|s| nn.forward(&s.to_owned())[0] * max_label).collect();
    let actuals: Vec<f64> = test.targets().iter().map(|&a| a * max_label).collect();
    let mse = predictions.iter().zip(actuals.iter()).map(|(p, a)| (p - a).powi(2)).sum::<f64>() / predictions.len() as f64;

    (predictions, actuals, mse)
}
