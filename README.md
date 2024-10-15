# Innovative Surface Roughness Detection Method Based on White Light Interference Images

Efficient and intelligent real-time roughness detection is essential for industrialized production of high-precision machining and manufacturing industries. The traditional optical inspection methods are designed for offline measurement which is not appropriate for online quick detection. The development of artificial intelligence and machine vision makes online roughness inspection possible. At present most methods utilize deep learning classification tasks, which can only provide one detection result under one field of view and cannot adapt to the inhomogeneous distribution of roughness. To address this issue, we propose a novel semantic segmentation approach processing with white light interference images to segment different roughness values under one field of view at the pixel level, thereby improving the detection results. The strategy and detection mechanism are expressed firstly. The strongest light intensity (0th level) interference fringe is the focus to be segmented. Based on this analysis, a semantic segmentation model, FAMNet, is proposed in this paper for improving surface roughness detection precision. White light interference image is as the input of model. Fusion Attention Mechanism (FAM) algorithm is specifically designed to be embedded into backbone to construct network structure. The channel attention and positional attention are combined together to achieve a balance between detection accuracy and computation time. The channel attention improves the ability of the attention network to perceive contextual information and enhances the connectivity between different channels. The positional attention mechanism reduces the loss of positional information and enhances the scope of utilization of network features. Comparison experiments are executed and the experiment results show that the fusion attention mechanism module algorithm has a better detection ability for white light interference fringe in the balance of accuracy and computing time. The segmentation recognition accuracy MIoU reaches 86.3%, and it improves the performance of computation by about 77.66 % compared to the current best-performing model and the amount of parameter is lower than it by about 63.59%. The proposed FAMNet model has a good performance for surface roughness online detection.


## Project Structure

```
project/
│
├── data_preprocess.py    # Data preprocess
├── FAMNet.py             # Model 
├── utils.py              # Utility functions
├── train.py              # Training script
├── test.py               # Testing script
├── requirements.txt      # Dependencies
├── checkpoints/          # Saved models
└── processed_dataset/    # Split and augmented data
    ├── train/
    ├── val/
    └── test/
```

## Dependencies

   ```bash
   pip install -r requirements.txt
   ```


## Notes

- **Data Consistency**: Ensure image and mask filenames match.
- **Normalization**: Images are normalized using ImageNet standards.
- **GPU Recommended**: For faster training and inference. 
- **Adjust Parameters**: Modify hyperparameters as needed for your dataset.

