from models.base_model import BaseModel
from transformers import BartTokenizer
from models.modeling_bart import BartForMultiModalGeneration
import evaluate

class BartMultiModal(BaseModel):

    def __init__(self, args):
        self.args = args
        super(BartMultiModal, self).__init__(args)
        
        # NEW: Get visual feature dimension (CLIP=512 or ResNet=2048)
        visual_hidden_size = getattr(args, 'visual_hidden_size', 2048)
        use_clip = getattr(args, 'use_clip', False)
        
        print("=" * 80)
        print("Initializing BartMultiModal")
        print(f"Visual feature dimension: {visual_hidden_size}")
        print(f"Using CLIP features: {use_clip}")
        print(f"Fusion layer: {args.fusion_layer}")
        print(f"Cross attention type: {args.cross_attn_type}")
        print("=" * 80)
        
        # NEW: Pass visual_hidden_size to the model
        self.model = BartForMultiModalGeneration.from_pretrained('facebook/bart-base',
                                                                 fusion_layer=args.fusion_layer,
                                                                 use_img_trans=args.use_img_trans,
                                                                 use_forget_gate=args.use_forget_gate,
                                                                 cross_attn_type=args.cross_attn_type,
                                                                 dim_common=args.dim_common,
                                                                 n_attn_heads=args.n_attn_heads,
                                                                 visual_hidden_size=visual_hidden_size)  # NEW!
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.rouge = evaluate.load('rouge')
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels, image_features, image_len):
        loss = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          labels=labels,
                          image_features=image_features,
                          image_len=image_len)[0]
        return loss

    def training_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids, image_features, image_len = batch
        # get loss
        loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids, image_features=image_features.float(), image_len=image_len)
        # logs
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids, image_features, image_len = batch
        # get summary
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                            image_features=image_features.float(),
                                            image_len=image_len)
        output = [summary_ids, label_ids]
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        summary = []
        reference = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, self.rouge)
        self.log('validation_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)
        self.save_txt(self.args.val_save_file, summary)
        self.save_txt(self.args.val_save_file+'reference', reference)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids, image_features, image_len = batch
        # get summary
        summary_ids = self.model.generate(input_ids=src_ids,
                                            attention_mask=mask,
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            early_stopping=True,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                            image_features=image_features.float(),
                                            image_len=image_len)
        output = [summary_ids, label_ids]
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        rouge = evaluate.load('rouge')
        summary = []
        reference = []
        for item in outputs:
            summary_id = item[0]
            label_id = item[1]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            one_reference = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_id]
            summary += one_summary
            reference += one_reference
        avg_rouge1, avg_rouge2, avg_rougeL = self.calrouge(summary, reference, rouge)
        self.log('test_Rouge1_one_epoch', avg_rouge1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge2_one_epoch', avg_rouge2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_RougeL_one_epoch', avg_rougeL, on_epoch=True, prog_bar=True, sync_dist=True)
        self.save_txt(self.args.test_save_file, summary)
        self.test_step_outputs.clear()

    def calrouge(self, summary, reference, rouge):
        final_results = rouge.compute(predictions=summary, references=reference)
        R1_F1 = final_results["rouge1"] * 100
        R2_F1 = final_results["rouge2"] * 100
        RL_F1 = final_results["rougeL"] * 100
        return R1_F1, R2_F1, RL_F1

    def save_txt(self, file_name, list_data):
        file = open(file_name, 'w')
        list_data = [item+'\n' for item in list_data]
        file.writelines(list_data)
        file.close()