"""
天气查询 Tool —— 扩展 BaseTool 的完整示例。

演示:
    1. 继承 BaseTool
    2. 设置类属性 name / description / parameters
    3. 实现 execute() 方法
    4. 在 main.py 中注册到 ToolRegistry

数据来源: wttr.in（免费、无需 API Key）
"""

import requests

from agent.tools.base import BaseTool, ToolParameter


class WeatherTool(BaseTool):
    name = "get_weather"
    description = (
        "查询指定城市的实时天气和未来天气预报。"
        "参数 city 可以是城市中文名（如'北京'）或英文名（如'Beijing'）。"
    )
    parameters = [
        ToolParameter(
            name="city",
            type="string",
            description="城市名称，中文或英文均可，例如：北京、Shanghai、东京",
        ),
    ]

    def execute(self, city: str) -> str:
        """调用 wttr.in 获取天气数据，返回可读文本。"""
        try:
            resp = requests.get(
                f"https://wttr.in/{city}",
                params={"format": "j1"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return self._format(data, city)
        except requests.RequestException as e:
            return f"[错误] 天气查询失败: {e}"
        except (KeyError, TypeError) as e:
            return f"[错误] 天气数据解析失败: {e}"

    # ---------- 私有 ----------

    @staticmethod
    def _format(data: dict, city: str) -> str:
        current = data.get("current_condition", [{}])[0]
        weather = data.get("weather", [])

        lines = [f"📍 {city} 天气"]
        lines.append(
            f"🌡 当前: {current.get('temp_C', '?')}°C  "
            f"体感: {current.get('FeelsLikeC', '?')}°C  "
            f"{current.get('weatherDesc', [{}])[0].get('value', '未知')}"
        )
        lines.append(
            f"💨 风速: {current.get('windspeedKmph', '?')} km/h  "
            f"💧 湿度: {current.get('humidity', '?')}%  "
            f"👁 能见度: {current.get('visibility', '?')} km"
        )

        lines.append("\n📅 未来预报:")
        for day in weather[:3]:
            date = day.get("date", "?")
            high = day.get("maxtempC", "?")
            low = day.get("mintempC", "?")
            desc = day.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", "未知")
            lines.append(f"  {date}  {desc}  ↑{high}°C / ↓{low}°C")

        return "\n".join(lines)
