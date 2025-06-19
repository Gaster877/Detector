## ! 須知
### 使用模型
目前使用[kgmann/ai-image-det-resnet18](https://huggingface.co/kgmann/ai-image-det-resnet18/tree/main)的模型作為判斷模型

### Dataset download link
*圖片資源大小皆為128x128，目前使用的Resnet18模型圖像輸入為224x224，並且做為測試用先取較小的當案大小，所以取128x128大小*

AI生成人像([OneMillionFaces](https://huggingface.co/datasets/RichardErkhov/OneMillionFaces?utm_source=chatgpt.com)):
用以訓練的AI人像資源來自OneMilionFaces的前100k筆資源，目前取其中的10k-20k作為測試
![image](https://github.com/user-attachments/assets/fb716d19-107b-4bea-8fc7-598a3909c00e)

真實人像([Flickr-Faces-HQ Dataset](https://github.com/NVlabs/ffhq-dataset))
用以訓練的AI人像資源來自FFHQ的70k筆資源，目前取其中的10k-20k作為測試
![image](https://github.com/user-attachments/assets/ea69a28e-fc3d-48bc-a5d9-b7ef52477007)

### 自主訓練模型(此處尚未使用上述資源訓練)
目前的模型準確率較低，需多加訓練
可直接載入使用
[Link here](https://drive.google.com/drive/folders/1zkMcR0KKC4zdEq2hsj-f0PU3iBL719eK?usp=sharing)
