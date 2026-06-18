# 1. 포워드 연산: 연산 기록(돌 놓기)
# 리니어 레이어를 통과시키면 파이토치는 연산 과정을 기록
output = model(input)

# 2. 동선 확인: grad_fn
# 결과값이 어떤 연산을 통해 나왔는지 확인
print(output.grad_fn)

# 3. 경로 추적: next_functions
# 계산 그래프의 역순으로 연결된 함수들(이전 노드들)을 확인
# 튜플 형태로 저장되어 있으며, 각 요소는 이전 연산 단계를 가리킵니다.
print(output.grad_fn.next_functions)

# 4. 역전파 실행: loss.backward()
# 기록된 동선(grad_fn)을 따라가며 미분값(gradient)을 계산하고 저장합니다.
loss.backward()

# 5. 결과 확인: .grad
# 이제 가중치마다 계산된 미분값이 .grad 속성에 저장됩니다.
print(weight.grad)

