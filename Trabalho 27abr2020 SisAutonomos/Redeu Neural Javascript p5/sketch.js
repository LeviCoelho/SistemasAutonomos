var train = true;

function setup() {
    createCanvas(500, 500);
    background(0);

    nn = new RedeNeural(2, 3, 1);

    // XOR Problem
    dataset = {
        inputs:
            [[1, 1],
            [1, 0],
            [0, 1],
            [0, 0]],
        outputs:
            [[0],
            [1],
            [1],
            [0]]
    }
}

function draw() {
    if (train) {
        epochs = 10000;
        for (var i = 0; i < epochs; i++) {
            var index = floor(random(4));
            nn.train(dataset.inputs[index], dataset.outputs[index]);
        }
        erroMinimo = 0.04;
        erroMaximo = 0.98;
        if (nn.predict([0, 0])[0] < erroMinimo && nn.predict([1, 0])[0] > erroMaximo) {//teste da rede neural
            train = false;
            console.log("terminou");
        }
    }
}