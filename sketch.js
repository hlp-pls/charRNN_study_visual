const lstm = ml5.charRNN('models/edgar_allan_poe_edited', modelReady);

let inputText = "";
let canvas;
let inputTextDom;
let modelStateDom;
let outputTextDom;
let start_pause_button;
let reset_button;
let generating = false;

let probs_n = [];
let probs_out = [];

let outputH_ = [];
let fullyConnectedWeights_ = [];
let weightedResult_ = [];
let logits_ = [];

let r_w;

let prediction_init = false;

function setup(){
    //console.log(tf);
    //let t_test = tf.tensor([1,2,3,4]);
    //let temp_tensor = tf.tensor(0.5);
    //let t_d_test = tf.div(t_test,temp_tensor);
    //console.log(t_d_test);

    modelStateDom = select('#modelState');
    modelStateDom.html('model loading...');
    inputTextDom = select('#inputText');
    inputText = inputTextDom.html();

    start_pause_button = createButton('run');
    start_pause_button.id('run_button');
    start_pause_button.mousePressed(generate);

    reset_button = createButton('reset');
    reset_button.id('reset_button');
    reset_button.mousePressed(resetModel);

    let cw = windowWidth-50;
    r_w = cw/114;
    canvas = createCanvas(cw,r_w*(4+256));

    outputTextDom = createSpan('');
    outputTextDom.id('outputText');
}

function draw(){

    background(0);
    
    if(prediction_init){
        noStroke();

        for(let i=0; i<256; i++){
            let c = outputH_[0][i];
            fill(255,255*c,0);
            rect(113*r_w,i*r_w,r_w,r_w);

            for(let j=0; j<113; j++){
                let index = j + j * i;
                let c2 = fullyConnectedWeights_[index];
                fill(255,0,255*c2);
                rect(r_w*j,r_w*i,r_w,r_w);
            }
        }

        translate(0,256*r_w);

        for(let i=0; i<113; i++){
            let c = weightedResult_[0][i];
            fill(c*255);
            rect(i*r_w,0,r_w,r_w);

            let c1 = logits_[0][i];
            fill(c1*255);
            rect(i*r_w,r_w*1,r_w,r_w);

            let c2 = probs_n[i];
            fill(c2*255);
            rect(i*r_w,r_w*2,r_w,r_w);

            let c3 = probs_out[i].probability;
            fill(c3*255);
            rect(i*r_w,r_w*3,r_w,r_w);
        }

    }
    
}

function modelReady(){
    console.log("LSTM model loaded.");
    modelStateDom.html('model loaded');
    resetModel();

    //console.log(lstm);
}

function resetModel(){
    lstm.reset();
    const seed = inputTextDom.html();
    lstm.feed(seed);
    outputTextDom.html(seed);
}

function generate(){
    if (generating) {
        generating = false;
        start_pause_button.html('run');
    } else {
        generating = true;
        start_pause_button.html('pause');
        loopRNN();
    }
}

async function loopRNN() {
  while (generating) {
    await predict();
  }
}

async function predict() {
    let temperature = 0.5;
  
    let next = await lstm.predict( temperature );

    await lstm.feed(next.sample);
    //console.log(next.probabilities);
    probs_out = next.probabilities;
    if(next.sample=="\n"){
        //console.log(next);
        outputTextDom.html("<br>",true);
    }
    outputTextDom.html(next.sample,true);
  
  
    let outputH = await lstm.state.h[1];
    outputH_ = outputH.arraySync();

    fullyConnectedWeights_ = await lstm.model.fullyConnectedWeights.data();
    //console.log(fullyConnectedWeights_);
    //console.log(outputH_);
    //console.log(lstm.state.c);
    let weightedResult = outputH.matMul(lstm.model.fullyConnectedWeights);
    weightedResult_ = weightedResult.arraySync();
    //console.log(weightedResult_);

    let logits = weightedResult.add(lstm.model.fullyConnectedBiases);
    logits_ = logits.arraySync();
    //console.log(logits_);
  
    let _temp = tf.scalar(0.5);
    //console.log(_temp);

    let logits_data = [];
    logits_data = logits.arraySync();

    let logits_copy = tf.tensor(logits_data);

    let divided = logits_copy.div(_temp);
    //console.log(divided);

    let _probabilities = tf.exp(divided);
    //console.log(_probabilities);
    let _probabilitiesNormalized = await tf.div(
        _probabilities,
        tf.sum(_probabilities),
    ).data();

    probs_n = _probabilitiesNormalized;
    //console.log(probs_n);
    prediction_init = true;
}