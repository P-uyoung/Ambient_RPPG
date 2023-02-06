<h2 align="center">Remote Heart Rate Monitoring</h2>

> **서울대학교 데이터사이언스대학원 2022 AI 경진대회**

> **원격 심박수 측정 프로젝트**

> **프로젝트 기간 : 2022.08.15 ~ 2022.09.18 (한달)**

> [데모영상](https://www.youtube.com/watch?v=GAX9GWvPWNs)

> [KCI 초록논문집,175p](http://conference21.kosombe.or.kr/register/2022_fall/file/ebook.pdf?v=230205)

<br/>

## **목차** 
<b>

- [요약](#요약)
- [기술 및 도구](#기술-및-도구)
- [구현](#구현)
  - [프로세스](#1-프로세스)
  - [데이터](#2-데이터)
  - [모델링](#3-모델링)
- [트러블 슈팅](#트러블-슈팅)
</b>
<br/>

## **요약**
- (input)영상의 RGB신호를 통해 (output)심박수 측정
- TensorflowLite 라이브러리를 사용하여 **모델 용량을 32.07% (2.34MB-> 0.75MB) 경량화하여** 서버 연동없이 스마트폰 기기만으로 원격 심박수 측정이 가능한 모델링 함.
<br/>

## **기술 및 도구**
  <span><img src="https://img.shields.io/badge/Python-05122A?style=flat-square&logo=python"/></span>
  <span><img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"></span>
  <span><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white"></span>
  <span><img src="https://img.shields.io/badge/TensorFlowLite-41454A?style=flat-square&logo=TensorFlowLite&logoColor=white"></span>
  <span><img src="https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=Linux&logoColor=white"></span>
  
<br/>


## **구현**
<details>
<summary><b>구현 설명 펼치기</b></summary>
<div markdown="1">

### 1. 프로세스
![](https://github.com/P-uyoung/Ambient_RPPG/blob/main/figure/method_process.png)

### 2. 데이터
  1. AFRL 데이터셋 : pre-train된 모델이 사용한 데이터  
  
  2. UBFC, VIPL-HR-V2 데이터셋 : 경량화한 모델에 대하여 성능 측정에 사용 [(일부 데이터)](https://github.com/P-uyoung/Ambient_RPPG/tree/main/VIPL_v2)  
<br/>

### 3. 모델링
- 모델링 결과
  1. UBFC 데이터셋 : 경량화에 따른 모델의 성능 변화(MAE)  
![](https://github.com/P-uyoung/Ambient_RPPG/blob/main/figure/UBFC_performance.png)   
  
    1. VIPL2 데이터셋 : 경량화에 따른 모델의 성능 변화(MAE)  
![](https://github.com/P-uyoung/Ambient_RPPG/blob/main/figure/VIPL2_performance.png)
  
- [코드 확인](https://github.com/P-uyoung/Ambient_RPPG/tree/main/ambient_rPPG/code)  
- 상세 설명   

  본 프로젝트 모델은 두 개의 모듈로 구성
  1. Face Detection 
      - 데이터 영상에서 얼굴에 해당하는 영역만 탐지하여 추출하는 모듈
      - residual neural network(ResNet)로 구현
      - ResNet은 기존의 CNN의 문제인 신경망의 레이어가 깊어질수록 데이터 학습이 어려워지는 문제를 해결한 알고리즘임.
  
  2. Heart Rate Estimation
      - 실시간 혈류량 변화에 따른 심박수 예측 모듈  
      - temporal shift-convolutional attention network(TS-CAN)로 구현
        1. Temporal shift (TF) 모듈  
        - 영상에서 특징 추출 시, 시간상의 정보(temporal information)를 고려하기 위해 3D convolution 연산해야 함.  
        ![](https://github.com/P-uyoung/Ambient_RPPG/blob/main/figure/TS.png)   
        
        <br/>
        
        2. CAN 모듈 : Attention mask 사용     
        - 이상치를 제거하여 얼굴 프레임에 더욱 초점을 맞춤.
        <br/>
        
       - 개념도  
       ![](https://github.com/P-uyoung/Ambient_RPPG/blob/main/figure/TS-CAN.png)    

</div>
</details>

</br>

## 트러블 슈팅
### 1. tensorflow(TF)로 pre-trained 된 모델 tensorflowLite(TFL)로 변환하는 과정에서 RuntimeError  
- 문제상황 : TFL는 TF와 달리, **batch inputs** 을 지원하지 않기 때문에 런타임 에러 발생함.

![](https://github.com/P-uyoung/Ambient_RPPG/blob/main/figure/Trouble_RuntimeError.png)   

- 해결 : batch를 input 배열의 차원으로 추가하여, 4차원 input으로 모델링 함.

<!--
<details>
<summary><b>기존 코드</b></summary>
<div markdown="1">

~~~java
/**
 * 게시물 Top10 (기준: 댓글 수 + 좋아요 수)
 * @return 인기순 상위 10개 게시물
 */
public Page<PostResponseDto> listTopTen() {

    PageRequest pageRequest = PageRequest.of(0, 10, Sort.Direction.DESC, "rankPoint", "likeCnt");
    return postRepository.findAll(pageRequest).map(PostResponseDto::new);
}

~~~

</div>
</details>
-->

