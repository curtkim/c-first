## main
    cmake-build-debug/bin/main

## infer_resnet

    python pytorch_model.py
    cmake-build-debug/bin/infer_resnet resnet50.onnx turkish_coffee.jpg
    
    ----------------------------------------------------------------
    Input filename:   resnet50.onnx
    ONNX IR version:  0.0.6
    Opset version:    9
    Producer name:    pytorch
    Producer version: 1.7
    Domain:           
    Model version:    0
    Doc string:       
    ----------------------------------------------------------------
    class: web site, website, internet site, site | confidence: 5.84855% | index: 916
    class: cleaver, meat cleaver, chopper | confidence: 3.78223% | index: 499
    ...