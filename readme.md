# 1.前言
最近文本生成图像AI太过于火爆，导致频频上热搜。
> **游戏设计师利用AI工具作画拿到一等奖：说的是美国的一位画师利用AI工具进行作画，并拿到了一等奖，从而惹来了大量的争议**

![image.png](https://cdn.nlark.com/yuque/0/2022/png/29330410/1662812463164-757d77af-f70a-44d7-8253-605730b87751.png#clientId=u2863d2b5-67ef-4&crop=0&crop=0&crop=1&crop=1&errorMessage=unknown%20error&from=paste&height=205&id=u6e2dbd7b&margin=%5Bobject%20Object%5D&name=image.png&originHeight=384&originWidth=1047&originalType=binary&ratio=1&rotation=0&showTitle=false&size=116234&status=error&style=none&taskId=u851cb537-8846-42c9-be99-d4c595533f3&title=&width=558.0037231445312)

> **由于AI图像生成软件Midjorunery的爆火，导致大量的日本画师纷纷进行抵制**

![image.png](https://cdn.nlark.com/yuque/0/2022/png/29330410/1662813510878-43588272-b54f-498b-b2da-4f06a246a2b1.png#clientId=u2863d2b5-67ef-4&crop=0&crop=0&crop=1&crop=1&errorMessage=unknown%20error&from=paste&height=214&id=ubbd9af83&margin=%5Bobject%20Object%5D&name=image.png&originHeight=461&originWidth=1083&originalType=binary&ratio=1&rotation=0&showTitle=false&size=133757&status=error&style=none&taskId=ub2fd09ad-6d49-429a-8a9e-eac016478ac&title=&width=503.00372314453125)

而且我之前也写过很多类似的文本生成图像模型，像Imagen和Dall.E2，都是我之前介绍过的作品：

那作为一个成功的“调包侠”，当然是要寻找有没有现成的工具包，可以让我们直接在本地电脑进行图像生成。这恰好Huggingface推出了这个扩散模型包**“Diffusers”**。
![image.png](https://cdn.nlark.com/yuque/0/2022/png/29330410/1664009010812-3666317c-9b9a-4602-a1f2-f989934149e4.png#clientId=ub65413d0-9df5-4&crop=0&crop=0&crop=1&crop=1&errorMessage=unknown%20error&from=paste&height=221&id=u582b2b39&margin=%5Bobject%20Object%5D&name=image.png&originHeight=365&originWidth=1278&originalType=binary&ratio=1&rotation=0&showTitle=false&size=91602&status=error&style=none&taskId=ua10d0bbf-4ca6-4fb0-8861-8e35cac144a&title=&width=774.5454097779354)
# 2.Diffusers
这个包有以下具体功能：

- 只需要几行代码，就能够利用扩散diffusion模型生成图像，简直是广大手残党的福音
- 可以使用不同的“噪声调节器”，来平衡模型生成速度和质量之间的关系
- 更有多种不同类型的模型，能够端到端的构建diffusion模型

要利用文本生成图片，主要有以下几个步骤：

- 安装对应的功能包
- 登陆huggingface网站，获取token
- 输入代码，下载模型，等待生成结构

## 1.1 功能包安装 + 获取Token
除了需要安装"Diffusers"之外：
```shell
pip install --upgrade diffusers
```

还需要安装“pytorch”，“transformers ”等
```shell
pip install transformers
```

其中pytorch安装方法，可以去官网根据自己的环境进行获取：
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
![image.png](https://cdn.nlark.com/yuque/0/2022/png/29330410/1664009929350-2382ba94-a8bc-4d63-8f3b-d86e7d2b7867.png#clientId=ub65413d0-9df5-4&crop=0&crop=0&crop=1&crop=1&errorMessage=unknown%20error&from=paste&height=359&id=u52e9bb53&margin=%5Bobject%20Object%5D&name=image.png&originHeight=941&originWidth=1354&originalType=binary&ratio=1&rotation=0&showTitle=false&size=133670&status=error&style=none&taskId=ub5b9d0c8-2269-4f29-bbe2-0f8f4226f88&title=&width=516.0037231445312)

除了安装python包之外，还需要去huggingface获取对应的token。

- 登陆官网，注册相应的账号，进行**settings**

![image.png](https://cdn.nlark.com/yuque/0/2022/png/29330410/1664010035251-1e0a2210-6a17-4b2d-b0ca-f82c59be9dcc.png#clientId=ub65413d0-9df5-4&crop=0&crop=0&crop=1&crop=1&errorMessage=unknown%20error&from=paste&height=374&id=u80757e42&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1018&originWidth=1346&originalType=binary&ratio=1&rotation=0&showTitle=false&size=212302&status=error&style=none&taskId=u588b126e-5708-48bd-9a76-a2c430b4419&title=&width=495.00372314453125)

- 新增自己token：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/29330410/1664010099587-0b30c5a9-00fc-447b-ba68-7c4e4a4bd11f.png#clientId=ub65413d0-9df5-4&crop=0&crop=0&crop=1&crop=1&errorMessage=unknown%20error&from=paste&height=330&id=uffec49f0&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1044&originWidth=1551&originalType=binary&ratio=1&rotation=0&showTitle=false&size=318335&status=error&style=none&taskId=udbdf8ad3-cee7-4056-9f27-f749b05c06a&title=&width=490.00372314453125)

- 在自己的命令行上，输入“huggingface-cli login”，出现successful说明成功

![image.png](https://cdn.nlark.com/yuque/0/2022/png/29330410/1664010156913-acc5c3a2-426d-43af-95cb-8fd392879ce7.png#clientId=ub65413d0-9df5-4&crop=0&crop=0&crop=1&crop=1&errorMessage=unknown%20error&from=paste&height=230&id=u31b34d42&margin=%5Bobject%20Object%5D&name=image.png&originHeight=379&originWidth=1415&originalType=binary&ratio=1&rotation=0&showTitle=false&size=20181&status=error&style=none&taskId=u38f8a5bd-f651-488f-ae87-1be15f90b41&title=&width=857.5757080092166)

## 1.2文本生成图像
这里直接调用最近很火的文本图像生成模型“Stable Diffusion”
```python
# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt).images[0]  
```

如果你想提前下载模型，然后进行加载，可以先执行下面命令：
```shell
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
```

然后重新执行代码：
```python
pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4")
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt).images[0]  
```
![astronaut_rides_horse.png](https://cdn.nlark.com/yuque/0/2022/png/29330410/1664011889980-e6d1ff11-97a4-44fb-a1fd-28db33077cbd.png#clientId=ub65413d0-9df5-4&crop=0&crop=0&crop=1&crop=1&errorMessage=unknown%20error&from=paste&height=310&id=u0b057d49&margin=%5Bobject%20Object%5D&name=astronaut_rides_horse.png&originHeight=512&originWidth=512&originalType=binary&ratio=1&rotation=0&showTitle=false&size=369131&status=error&style=none&taskId=uabe9ac92-b430-4748-9373-60e8c47497f&title=&width=310.3030123679992)

## 1.3 指导图像生成
不仅仅可以从0开始生成一张图片，Diffusers可以利用现有的一张图片，加入自己的语言进行指导，然后重新生成一张图片。

比如原始图片：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/29330410/1664012648903-b09fe916-12e8-4964-b6af-a1261c5f6c75.png#clientId=ub65413d0-9df5-4&crop=0&crop=0&crop=1&crop=1&errorMessage=unknown%20error&from=paste&height=248&id=u9fc1696e&margin=%5Bobject%20Object%5D&name=image.png&originHeight=773&originWidth=1539&originalType=binary&ratio=1&rotation=0&showTitle=false&size=632959&status=error&style=none&taskId=ud6d1dd18-5815-461e-8538-57702b2e958&title=&width=493.998046875)

加入语言进行指导，让它生成更加艺术性
> A fantasy landscape, trending on artstation

最后生成图片如下：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/29330410/1664013766796-1f414104-dac6-47c0-8417-302a9b9b29bc.png#clientId=ud764a777-f6ca-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=310&id=uf41f16d9&margin=%5Bobject%20Object%5D&name=image.png&originHeight=512&originWidth=768&originalType=binary&ratio=1&rotation=0&showTitle=false&size=554045&status=done&style=none&taskId=u3f26594c-ea51-4e72-8ed3-1a56be287c1&title=&width=465.4545185519988)
那这种可玩性就更高了，由此扩展，是不是给定一张有水印的图片，就可以生成无水印的呢？或者更进一步（自行脑补三千字），咳咳~。再说下去，可能就要被封了，哈哈哈。

本次教程就到这里拉，我是leo~，欢迎关注我的公众号/知乎"算法一只狗"，我们下期再见~
![qrcode_for_gh_f4f620aeff8d_258.jpg](https://cdn.nlark.com/yuque/0/2022/jpeg/29330410/1664013144190-7f3e8a49-0bc9-47d6-aa55-914006b77efb.jpeg#clientId=ud764a777-f6ca-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=156&id=ua17f0f15&margin=%5Bobject%20Object%5D&name=qrcode_for_gh_f4f620aeff8d_258.jpg&originHeight=258&originWidth=258&originalType=binary&ratio=1&rotation=0&showTitle=false&size=27597&status=done&style=none&taskId=u840c7fb5-a16c-47e4-843d-0a0b7294ada&title=&width=156.3636273260621)

#   S t a b l e D i f f u s i o n U s e  
 