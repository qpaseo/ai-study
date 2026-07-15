**개념**

1. Diffusion [Diffusion](./Diffusion.md)
2. Transformer [Transformer](./Transformer.md)
3. Layer Normalization [Layer Normalization](./Layer_Normalization.md)

**개요**
- Diffusion을 할때 딥러닝 역할을 Transformer처럼해서 Diffusion을 구현하는것

**왜? Transformer인가**

- Transformer
  - _스케일링 법칙_
    - 이전의 U-Net(CNN) 구조는 이미지 품질을 올릴수 있는 한계치가 있었음
    - 근데 Transformer 방식은 자원을 많이 넣으면 우상향 형식으로 성능이 좋아짐
  - _한번에 넓은 곳_
    - 이전의 U-Net(CNN) 구조는 이미지를 3, 3씩 천천히 지나가는 방식이였는데 이럼 왼쪽, 오른쪽 끝이 잘 연관되지 못하는 현상이 있었음
    - Transformer을 사용하면 한번에 펼쳐놓고 관계를 확인할수 있기 현상이 줄어듬
      - 언어 Transformer에서는 뒤에 언어를 가릴 필요가 없기 때문에 한번에 모든 출력을 사용할수 있어서
        ("안녕하세요 저는 ai입니다" 에서 "안녕하세요 저는" 뒤에 "ai입니다" 를 추론하기위해서는 "저는" 뒤에를 가려야 추론할수 있는데 이미지 만들때에는 가릴 필요가 없어서)
  - _강력한 유연성_
    - Transformer을 사용하면 이미지를 시간정보, 프롬프트에 따른 구조를 함께 조합해서 노이즈에서 이미지를 만들때
      이미지 조각 정보 사이에 텍스트 정보와 시간 정보를 끼워 넣는 연산에 특화가 되어있음

**왜? 전에는 U-Net(CNN)을 사용했는가?**

- 이미지의 디테일한 위치 정보를 보존하는 데 탁월하고 컴퓨터 사양을 덜 잡아먹기 때문에 사용했었음
- Transformer 방식이 메모리를 많이 사용했는데 이제는 cpu와 같은 하드웨어의 성능이 좋아져서 사용 가능함

**Transformer를 사용하는데 달라진점은?**

- _encoder only 구조_
  - 1. 마스킹이 필요없음 (자세한 설명은 19번줄 참고)
  - 2. 입력, 출력 크기가 동일함
  - => 그래서 decoder는 사용하지 않고 encoder만 사용함
- _time embedding 추가_
  - t : 이미지에서 노이즈를 걷어네는 과정에서 과정과 동기화 되어있는 값으로 0이면 노이즈 1이면 최종 이미지로 생각하는 값
  - 이 t를 모델에 같이 넣어줘서 "지금 노이즈에서 이미지까지 어느정도 진행되었구나" 혹은 "현제 이미지에 노이즈가 어느정도 포함되었구나"를 알수 있게 하도록 함
- _conditioning 추가_ 
  - 사용자가 입력한 text(프롬프트)를 이미지 만들때에 참고해서 넣어주는것
  - dit에서 특히 중요함
  - 추가 방법
    - 1. condition도 데이터의 일부처럼 사용
      - 데이터와 함께 condition도 줌 (이미지랑 condition을 단순히 뒤에 붙여서 넣는방법)
      - 출력에서 이미지 데이터랑 condition에 대한 출력도 주는데 출력된 condition는 버림
      - 실제로 거의 사용하지 않음
    - 2. condition을 key value형식으로 사용하는것
      - 데이터랑 별게로 condition을 제공 (참고서 느낌)
      - 1번보다 성능이 좋다고는 하더라
    - 3. adaLN (Adaptive Layer Normalization)
      - Layer Normalization와 달리 condition에 따라 스케일, 시프트를 변경함
      - 한계점
        - Single Label는 잘함
          - 고양이, 자동차, 사람과 같이 한가지의 조건이 주어지는것만 가정함
      - 한계점 해결방법
        - 다른 모델과 같이 사용함, 긴 조건(프롬프트)는 key value형식으로 처리하고, time embedding은 adaLN로 처리함
    - 4. adaLN-Zero
      - adaLN에 스케일을 조절해주는 방법이 생긴 버전
        - 계산된 값에 스케일값을 곱해주는 연산을 추가함
      - Zero인 이유
        - 스케일값을 초기에 0으로 설정해서
          - 초기에 불안정한 데이터에 스케일연산을 하지않고 안정적이게 된 데이터가 된 뒤부터 스케일 연산을 하기위해