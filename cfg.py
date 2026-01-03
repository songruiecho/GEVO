class Config:
    def __init__(self):
        # self.LLM_path = "/home/chenhechang/models/"
        self.LLM_path = "/home/lyj/models/"
        self.VLM = 'GLM-4.1V-9B-Thinking'   # InternVL3_5-1B-HF   Qwen3-VL-2B-Instruct  GLM-4.1V-9B-Thinking
        self.task = 'task1'
        self.encoder = 'clip-vit-base-patch32'
        self.batch_size = 32
        self.shots = 0