ViT(Vision Transformer)
- 이미지를 patch 로 쪼갠다
  - 이미지를 pixel로 쪼개는게 아니라 **의미단위** 취급을 위해 patch로 만듬
  - patch는 더 많은 정보를 담고 있으므로 attention(정렬)이 가능함
  => 그래서 배열로 입력받음
- Transformer로 이미지를 이해한다
  - patch로 분리된 이미지를 입력받아 이미지 끼리 상관관계를 이해하고 파악함
- CLS token의 역할
  - 존재 이유 : 새상에 사진은 많고 규격은 전부 다르기 때문에 출력이나 입력이 동일하지 않을 가능성이 있음 그래서 patch로 나누고 Transformer로 이미지를 이해한 결과를 CLS토론으로 만드는 과정을 거치는데
  - 