function setup() {
    createCanvas(400, 400);
    background(220);

    var m1 = new Matrix(2, 2);
    var m2 = new Matrix(2, 2);
    //Matrix.add(m1,m2).print();
    var m3 = new Matrix(2, 1);
    var m4 = new Matrix(1, 2);
    //Matrix.multiply(m1,m2).print();


    var rn = new RedeNeural(1, 3, 1);
    var arr = [1, 2];
    rn.feedforward(arr);

}

function draw() {

}
