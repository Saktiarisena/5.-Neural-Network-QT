use qmetaobject::*;
use crate::neural_net::SimpleNeuralNetwork;

#[derive(QObject, Default)]
pub struct RiceApp {
    base: qt_base_class!(trait QObject),

    #[qt_property(f64)]
    pub learning_rate: f64,

    #[qt_property(i32)]
    pub epochs: i32,

    #[qt_property(f64, notify(mse_changed))]
    pub mse: f64,

    #[qt_signal]
    pub mse_changed: Signal,
}

impl RiceApp {
    #[qt_method]
    pub fn train(&mut self) {
        let path = "data/Rice_MSC_Dataset_sample.csv";
        let (_preds, _actuals, mse) = 
            SimpleNeuralNetwork::train_and_predict(path, self.learning_rate, self.epochs as usize);

        self.mse = mse;
        self.mse_changed.emit();
    }
}
