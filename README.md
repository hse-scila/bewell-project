# BeWell
The repo contains materials for the project: *"Predicting subjective well-being in a high-risk sample of Russian mental health app users"*

### Keywords: 
 - digital traces;
 - subjective well-being;
 - mental health prediction
## Project Description 
Despite recent achievements in predicting personality traits and some otherhuman psychological features with digital traces, prediction of subjective well-being (SWB) appears to be a relatively new task with few solutions. COVID-19 pandemic has added both a stronger need for rapid SWB screening and new opportunities for it, with online mental health applications gaining popularity and accumulating large and diverse user data. Nevertheless, the few existing works so far have aimed at predicting SWB only in terms of Diener’s Satisfaction with Life Scale. None of them analyzes the scale developed by the World Health Organization, known as WHO-5 – a widely accepted tool for screening mental well-being and, specifically, for depression detection. Moreover,existing research is limited to English-speaking populations, and tend to use text,network and app usage types of data separately. In the current work, we cover these gaps by predicting both mentioned SWB scales on a sample of Russian mental health app users who represent a population with high risk of mental health problems. In doing so, we employ a unique combination of phone application usage data with private messaging and networking digital traces from VKontakte, the most popular social media platform in Russia. As a result, we predict Diener’s SWB scale with the state-of-the-art quality, introduce the first predictive models for WHO-5, with similar quality, and reach high accuracy in the prediction of clinically meaningful classes of the latter scale. Moreover, our feature analysis sheds light on the interrelated nature of the two studied scales: they are both characterized by negative sentiment expressed in text messages and by phone application usage in the morning hours, confirming some previous findings on subjective well-being manifestations. At the same time, SWB measured by Diener’s scale is reflected mostly in lexical features referring to social and affective interactions, while mental well-being is characterized by objective features that reflect physiological functioning, circadian rhythms and somatic conditions, thus saliently demonstrating the underlying theoretical differences between the two scales.

## Repo Description
Two folders contain scripts and notebooks connected with project results. 
Unfortunately you cannot easily reproduce results because datasets are not publicly available. As it is mentioned in the paper, the data is not publicly available, because it contradicts the terms and conditions under which the data were collected from `DigiFreud` users (privacy policy).
When onboarding in the `Digital Freud app`, users agree to a privacy policy that does not imply the publication of the dataset or its transfer to third parties. Largely, because the amount of data for each user does not allow to completely anonymize the dataset.
They contained sensitive personal information before anonymization, but still they. 
However, you can apply ready-to-use pipelines for classification and regression task using simple models like linear regression or more complicated based on pretrained BERT.

Main folders:
- `notebooks/clf`
- `notebooks/regression`

Using this python code we obtained result reported in the paper.

### Authors: 
- Polina Panicheva (ppolin86@gmail.com)
- Larisa Mararitsa  (larisamararitsa@mail.ru)
- Semen Sorokin (sorokin.semen2020@gmail.com)
- Olessia Koltsova (ekoltsova@hse.ru)
- Paolo Rosso (prosso@dsic.upv.es)
