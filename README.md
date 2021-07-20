# Seq2Seq_Radar_Echo_ConvLSTM

## Get Started Train the ConvLSTM model
1. Install env_v2

2. Set parameters 

check convlstm_train.py is in the 'save_path' folder

check load_radar_echo_df_path is in the 'save_path' folder

check model name
```
parser.add_argument('--model_name', type=str, default='convlstm')
```

You can set pretrained model and check pretrained_model is in the 'save_path' folder

You can set parameters, ex: weighted loss function

```
core/models/model_factory_LayerNormpy.py /

self.weight = [1,2,5,10,30,40]

self.custom_criterion = MyMSELoss(self.weight)
```

3. Train the model

1. Set parameters 

```
python -m convlstm_train.py
```

## Test the ConvLSTM model

check convlstm_test.py is in the 'save_path' folder

check model name
```
parser.add_argument('--model_name', type=str, default='convlstm')
```
check model name, ex:

```
model_name = 'model_itr20_test_cost18.401666419122662.pkl'
```

2. Test and evaluation

```
python -m convlstm_test.py
```
