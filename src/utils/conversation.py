import re
import dataclasses
from enum import Enum, auto
from typing import List

class Version(Enum):
    QWEN2 = auto()
    DEEPSEEK2 = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""                                          

    system: str
    roles: List[str]
    messages: List[List[str]]
    sep: str = "###"
    version: str = "Unknown"
    skip_next: bool = False
    func_pattern: str = ""
    template_type: str = "qwen"

    def get_prompt(self):
        rets, is_target = self.get_prompt_split_by_target()
        ret = "".join(rets)
        return ret

    def get_prompt_split_by_target(self):
        messages = self.messages
        # return prompts as:
        # [
        #   "<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n,
        #   "{message}<|im_end|>",
        #   "\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n",
        #   "{message}<|im_end|>",
        #   "\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n",
        #   "{message}<|im_end|>",
        #   ...
        #   "\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n",
        #   "{message}<|im_end|>",
        # ]
        if self.template_type == "qwen":
            wrap_qa = lambda msg: f"<|im_start|>{msg}<|im_end|>\n"
            rets = [wrap_qa(f"system\n{self.system}")]
            is_target = [False]
            # assert len(messages) % 2 == 0, len(messages)
            role = ""
            for i, (role, message) in enumerate(messages):
                assert messages, f"message: {message} is empty"
                if type(message) is tuple:
                    message, _, _ = message
                
                head = f"<|im_start|>{role}\n"
                body = f"{message}<|im_end|>"
                tail = f"\n"
                if role == "assistant":
                    rets[-1] += head
                    rets.append(body)
                    rets.append(tail)
                    is_target.extend([True, False])
                elif role == "user" or role == "observation":
                    rets[-1] += f"{head}{body}{tail}"
                else:
                    raise ValueError(f"{role} not implemented")
            if role == "assistant":
                rets, is_target = rets[:-1], is_target[:-1]
        elif self.template_type == "deepseek":
            '''
            <｜begin▁of▁sentence｜>{system_message}

            User: {user_message_1}

            Assistant: {assistant_message_1}<｜end▁of▁sentence｜>
            
            User: {user_message_2}

            Assistant: {assistant_message_2}<｜end▁of▁sentence｜>
            
            User: {user_message_3}

            Assistant: {assistant_message_3}<｜end▁of▁sentence｜>User: {user_message_2}

            Assistant:
            '''
            rets = [ f"<｜begin▁of▁sentence｜>{self.system}\n\n" ]
            is_target = [False]
            role = ""
            for i, (role, message) in enumerate(messages):
                assert messages, f"message: {message} is empty"
                if type(message) is tuple:
                    message, _, _ = message
                
                body = f"{message}<｜end▁of▁sentence｜>"
                tail = "\n\n"
                
                if role == "assistant":
                    rets[-1] += "Assistant: "
                    rets.append(body)
                    rets.append(tail)
                    is_target.extend([True, False])
                elif role == "user" or role == "observation":
                    rets[-1] += role[0].upper() + role[1:] + f": {message}\n\n" 
                else:
                    raise ValueError(f"{role} not implemented")
            if role == "assistant":
                rets, is_target = rets[:-1], is_target[:-1]
        else:
            raise ValueError(f"{self.template_type} chat template not implemented!")
            
        return rets, is_target

    def has_function_call(self, message):
        has_func_call = re.search(self.func_pattern, message) is not None
        return has_func_call
    
    def remove_function_call(self, message):
        has_func_call = self.has_function_call(message)
        if has_func_call:
            ret, = re.match(f"(.*)(?={self.func_pattern})", message.strip()).groups()
            ret = f"{ret}"
            return ret
        return message

    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            sep=self.sep,
            version=self.version,
            func_pattern=self.func_pattern,
            template_type=self.template_type,
        )

conv_qwen2 = Conversation(
    system="You are an AI robot and your name is Lucy. \n"
        "- You are a multimodal large language model developed by Tencent. Your aim is to be helpful, honest and harmless. \n"
        "- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n"
        "- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user.",
    roles=("user", "assistant", "observation"),
    version=Version.QWEN2,
    messages=(),
    sep="<|im_start|>",
    func_pattern="<function=.*</function>",
)

conv_deepseek = Conversation(
    system="You are an AI robot and your name is Lucy. \n"
        "- You are a multimodal large language model developed by Tencent. Your aim is to be helpful, honest and harmless. \n"
        "- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n"
        "- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user.",
    roles=("user", "assistant", "observation"),
    version=Version.DEEPSEEK2,
    messages=(),
    sep="<｜end▁of▁sentence｜>",
    func_pattern="<function=.*</function>",
    template_type="deepseek",
)
