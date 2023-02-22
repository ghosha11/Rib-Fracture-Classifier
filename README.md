# Rib-Fracture-Classifier
ResNet and DesneNet Classifiers to identify fractures on frontal radiographs of children under 2 years of age.

# Deep Learning-based prediction of rib fracture presence in frontal radiographs of children under two years of age: A proof of concept study
Objectives:

In this proof-of-concept study, we aimed to develop deep-learning-based classifiers to identify rib fractures on frontal chest radiographs in children under two years of age.

Methods:

This retrospective study included 1311 frontal chest radiographs (radiographs with rib fractures, n = 653) from 1231 unique patients (median age: 4 m). Patients with more than one radiograph were included only in the training set. A binary classification was performed to identify the presence or absence of rib fractures using transfer learning and Resnet-50 and DenseNet-121 architectures. The area under the receiver operating characteristic curve (AUC-ROC) was reported. Gradient-weighted class activation mapping was used to highlight the region most relevant to the deep learning models’ predictions.
Results:

On the validation set, the ResNet-50 and DenseNet-121 models obtained an AUC-ROC of 0.89 and 0.88, respectively. On the test set, the ResNet-50 model demonstrated an AUC-ROC of 0.84 with a sensitivity of 81% and specificity of 70%. The DenseNet-50 model obtained an AUC of 0.82 with 72% sensitivity and 79% specificity.
Conclusions:

In this proof-of-concept study, a deep learning-based approach enabled the automatic detection of rib fractures in chest radiographs of young children with performances comparable to pediatric radiologists. Further evaluation of this approach on large multi institutional datasets is needed to assess the generalizability of our results.
Advances in knowledge:

In this proof-of-concept study, a deep learning-based approach performed well in identifying chest radiographs with rib fractures. These findings provide further impetus to develop deep learning algorithms for identifying rib fractures in children, especially those with suspected physical abuse or non-accidental trauma.
