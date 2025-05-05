import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    width: 400
    height: 300
    visible: true
    title: "Rice Classifier Qt5"

    Column {
        spacing: 10
        padding: 20

        Text { text: "Learning Rate: " + app.learning_rate }
        Slider {
            from: 0.01; to: 1.0; value: app.learning_rate
            onValueChanged: app.learning_rate = value
        }

        Text { text: "Epochs: " + app.epochs }
        Slider {
            from: 100; to: 5000; value: app.epochs
            onValueChanged: app.epochs = value
        }

        Button {
            text: "Train Model"
            onClicked: app.train()
        }

        Text {
            text: "MSE: " + app.mse.toFixed(4)
        }
    }
}
