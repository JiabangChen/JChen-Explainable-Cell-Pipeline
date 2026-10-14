# JC-Cell-Pipeline

This is the code for paper "coming soon". Besides, we have published the cancer cell and normal cell image database, you can access it freely via "coming soon". This database is open access and continuing update, please cite this paper and the database if you want to use our database.

Citation for paper "xxxxxxx"
Citation for database "xxxxxxxx"

This research is funded by the University of Sheffield Scholarships.

This work aims to generate a deep learning based and automatic pipeline to recognition cancer cells and normal cells from whole-slide fluorescent microscopy images with high-throughput manner. Specifically, we use a pre-trained and domain-generalised cell segmentation model to identify each single cell from large microscopy images without extra human annotations and model training. Then, we use self-supervised models: DINO-based models, such as DINO, DINOv2, and DINOv3, to classify these images (each contains a single cell), which demonstrates the strong predictability. Finally, we use HiResCAM as the XAI method to explain the classification model. The highlighted regions in the model decision-making process are aligned well with the biologists' knowledge, which proves the trustworthiness of our pipeline and is very crucial in real clinical application. In addition, we also reveal that we can only use the cytoskeleton in the cytoplasm to classify the cancer and normal cells, which explore the diagnosis potential of cytoskeleton-based markers. We believe our novel pipeline can help the biologists to screen some cancer suppressing or reversing drugs by observing the behavior of cancer cells under different drugs. This work paves the way for further cancer diagnosis and treatment, especially for some uncommon cancers and bespoke therapies development.

# Contact details and Job description:
Jiabang Chen: jchen144@sheffield.ac.uk, Ameer Alwadiya: ameer.alwadiya@outlook.com, Annica Gad: annica.gad.2@ki.se / annica.gad@oru.se, Antonija Kezic: antonija.kezic.el@gmail.com, and George Panoutsos: g.panoutsos@sheffield.ac.uk.

Jiabang Chen is response for all technical parts including image pre-processing, segmentation, classification, XAI, and experiment design, and write the whole paper. He is the corresponding author.

Ameer Alwadiya realize the technical parts of cell segmentation and write the paragraph of cell segmentation in the literature review, methodology. and experiment implementation.

Dr. Annica Gad and Antonija Kezic are in charge of cell culturing, staining, photo taking, and writing all biological-related parts in the paper.

Prof. George Panoutsos is the project leader and supervisor.

# Note:
PLease refer to the link: https://github.com/facebookresearch/dinov3 for more information about the weights of DINOv3 and how to implement it.
