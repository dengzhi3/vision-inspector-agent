from pydantic import BaseModel


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


item = DetectionItem(
    class_id=0,
    class_name="crack",
    confidence=0.91,
    bbox=[10, 20, 300, 400],
)

print(item)
print(type(item))
print(item.model_dump())