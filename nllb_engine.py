from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class TranslatorNLLB:
    def __init__(self):
        # Utilisation de la version distillée pour la rapidité sur PC
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def translate(self, text, src_code, tgt_code):
        if not text or len(text.strip()) == 0:
            return ""
        
        # Pipeline de traduction
        translator = pipeline(
            'translation', 
            model=self.model, 
            tokenizer=self.tokenizer, 
            src_lang=src_code, 
            tgt_lang=tgt_code, 
            max_length=1024
        )
        output = translator(text)
        return output[0]['translation_text']