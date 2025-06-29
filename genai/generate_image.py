"""
Personalized City Poster Generator

This module generates personalized city posters that reflect the characteristics of the city,
the theme of the day, AQI information, and health information using either:
1. Amazon Bedrock's image generation capabilities
2. Locally deployed Stable Diffusion models via Hugging Face
3. Simple local image generation (fallback)

The generated poster combines:
1. A city-specific image showing the current air quality conditions
2. AQI information overlay
3. Health recommendations based on the AQI level
"""

import os
import json
import boto3
import base64
import logging
import requests
import platform
import subprocess
from io import BytesIO
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from botocore.exceptions import ClientError, NoRegionError
from tqdm.auto import tqdm

# Import prompt templates
from .prompt_templates import (
    generate_city_poster_prompt,
    get_aqi_category,
    get_health_advice
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for Diffusers
DIFFUSERS_AVAILABLE = False
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
CUDA_AVAILABLE = False

# Try to import optional dependencies
try:
    import torch
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    from diffusers.utils import is_accelerate_available
    from huggingface_hub import hf_hub_download, HfFolder
    from tqdm.auto import tqdm
    
    TORCH_AVAILABLE = True
    DIFFUSERS_AVAILABLE = True
    
    # Check for MPS (Apple Silicon GPU) support
    if torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        logger.info("Apple Silicon MPS acceleration is available")
    
    # Check for CUDA support
    if torch.cuda.is_available():
        CUDA_AVAILABLE = True
        logger.info("CUDA acceleration is available")
        
except ImportError:
    logger.warning("Hugging Face Diffusers and/or PyTorch not available. Install with: pip install torch diffusers transformers accelerate huggingface_hub tqdm")

def get_mac_hardware_info():
    """Get information about hardware for model selection across different operating systems."""
    model = "Unknown"
    cpu_info = "Unknown"
    memory_gb = 8  # Default to 8GB if we can't detect
    is_apple_silicon = False
    
    try:
        system = platform.system()
        
        # macOS specific hardware detection
        if system == "Darwin":
            # Get Mac model
            model = subprocess.check_output(["sysctl", "-n", "hw.model"]).decode("utf-8").strip()
            
            # Get CPU info
            cpu_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8").strip()
            
            # Get memory info (in bytes, convert to GB)
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode("utf-8").strip())
            memory_gb = mem_bytes / (1024 ** 3)
            
            # Check if Apple Silicon
            is_apple_silicon = "Apple" in cpu_info
            
            logger.info(f"Mac hardware: {model}, CPU: {cpu_info}, Memory: {memory_gb:.1f} GB")
            
        # Linux specific hardware detection
        elif system == "Linux":
            # Get CPU info
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_info = line.split(":")[1].strip()
                            break
            except:
                pass
                
            # Get memory info
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal" in line:
                            # MemTotal is in kB, convert to GB
                            memory_kb = int(line.split()[1])
                            memory_gb = memory_kb / (1024 * 1024)
                            break
            except:
                pass
                
            # Try to get model info
            try:
                with open("/sys/devices/virtual/dmi/id/product_name", "r") as f:
                    model = f.read().strip()
            except:
                pass
                
            logger.info(f"Linux hardware: {model}, CPU: {cpu_info}, Memory: {memory_gb:.1f} GB")
            
        # Windows specific hardware detection
        elif system == "Windows":
            try:
                # Get CPU info using wmic
                cpu_info = subprocess.check_output("wmic cpu get name", shell=True).decode("utf-8").strip()
                cpu_info = cpu_info.split("\n")[1].strip()
            except:
                pass
                
            try:
                # Get memory info using wmic
                mem_output = subprocess.check_output("wmic ComputerSystem get TotalPhysicalMemory", shell=True).decode("utf-8").strip()
                mem_bytes = int(mem_output.split("\n")[1].strip())
                memory_gb = mem_bytes / (1024 ** 3)
            except:
                pass
                
            try:
                # Get model info
                model = subprocess.check_output("wmic csproduct get name", shell=True).decode("utf-8").strip()
                model = model.split("\n")[1].strip()
            except:
                pass
                
            logger.info(f"Windows hardware: {model}, CPU: {cpu_info}, Memory: {memory_gb:.1f} GB")
            
        # Fallback to psutil if available
        if memory_gb == 8:
            try:
                import psutil
                memory_gb = psutil.virtual_memory().total / (1024 ** 3)
                logger.info(f"Detected {memory_gb:.1f} GB memory using psutil")
            except ImportError:
                logger.warning("psutil not available for memory detection")
                
    except Exception as e:
        logger.warning(f"Error getting hardware info: {str(e)}")
    
    return {
        "model": model,
        "cpu": cpu_info,
        "memory_gb": memory_gb,
        "is_apple_silicon": is_apple_silicon
    }


class CityPosterGenerator:
    """
    A class to generate personalized city posters with AQI information and health advice.
    """
    
    def __init__(self, model_id="stability.stable-diffusion-xl", s3_bucket=None, mode="auto", 
                 local_model_url="http://localhost:7860"):
        """
        Initialize the CityPosterGenerator.
        
        Args:
            model_id (str): The Bedrock model ID to use for image generation
            s3_bucket (str, optional): S3 bucket name for storing generated images
            mode (str): Generation mode - "auto", "local", "local_sd", or "bedrock"
                - "auto": Try Bedrock first, fall back to local_sd if available, then local fallback
                - "local": Use simple local image generation (no ML model)
                - "local_sd": Use locally deployed Stable Diffusion model via Hugging Face
                - "bedrock": Use AWS Bedrock only
            local_model_url (str): URL for the local Stable Diffusion API endpoint (if using API)
        """
        self.mode = mode
        self.model_id = model_id
        self.s3_bucket = s3_bucket
        self.local_model_url = local_model_url
        # 使用 stabilityai/sd-turbo 模型，速度更快
        self.local_model_id = "stabilityai/sd-turbo"
        self.bedrock_client = None
        self.s3_client = None
        self.sd_pipeline = None
        
        # Initialize AWS clients if not in local-only modes
        if mode not in ["local", "local_sd"]:
            try:
                self.bedrock_client = boto3.client('bedrock-runtime')
                self.s3_client = boto3.client('s3') if s3_bucket else None
                logger.info("AWS Bedrock client initialized successfully")
            except (ClientError, NoRegionError) as e:
                logger.warning(f"AWS client initialization failed: {str(e)}.")
                if mode == "bedrock":
                    raise ValueError("Bedrock mode requested but AWS client initialization failed")
                self.bedrock_client = None
                self.s3_client = None
        
        # Check for Hugging Face Diffusers availability
        self.diffusers_available = DIFFUSERS_AVAILABLE and TORCH_AVAILABLE
        
        # Initialize Stable Diffusion pipeline if needed
        if mode in ["auto", "local_sd"] and self.diffusers_available:
            try:
                self._initialize_diffusion_pipeline()
                logger.info("Hugging Face Diffusers pipeline initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Diffusers pipeline: {str(e)}")
                if mode == "local_sd":
                    raise ValueError(f"Local SD mode requested but initialization failed: {str(e)}")
        
        # Determine the actual mode based on availability
        if self.bedrock_client:
            self.actual_mode = "bedrock"
        elif self.diffusers_available and self.sd_pipeline is not None:
            self.actual_mode = "local_sd"
        else:
            self.actual_mode = "local"
            
        logger.info(f"CityPosterGenerator initialized in {self.actual_mode} mode")
    
    def _initialize_diffusion_pipeline(self):
        """Initialize the Stable Diffusion pipeline based on hardware capabilities."""
        if not self.diffusers_available:
            logger.warning("Diffusers not available, skipping pipeline initialization")
            return
        
        # 使用 stabilityai/sd-turbo 模型，速度更快
        logger.info(f"Using model: {self.local_model_id}")
        print(f"使用高速模型: {self.local_model_id}")
        
        # Get hardware info for memory optimizations
        hw_info = get_mac_hardware_info()
        memory_gb = hw_info["memory_gb"]
        is_apple_silicon = hw_info["is_apple_silicon"]
        
        # Determine device
        if CUDA_AVAILABLE:
            device = "cuda"
        elif MPS_AVAILABLE and is_apple_silicon:
            device = "mps"
        else:
            device = "cpu"
            
        logger.info(f"Using device: {device}")
        print(f"使用设备: {device}")
        
        # Load the pipeline with appropriate settings
        try:
            print("正在加载模型到内存，请耐心等待...")
            
            # 使用统一的加载方式，简化代码
            torch_dtype = torch.float32
            if device == "cuda":
                torch_dtype = torch.float16
            
            # 加载 sd-turbo 模型
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                self.local_model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,  # 禁用安全检查以提高性能
                requires_safety_checker=False
            )
            
            # 移动到适当的设备
            if device != "cpu":
                print(f"正在将模型移动到 {device}...")
                if device == "mps":
                    # 对于 MPS，只移动 UNet 到 GPU
                    self.sd_pipeline.unet.to(device)
                else:
                    # 对于 CUDA，移动整个模型
                    self.sd_pipeline = self.sd_pipeline.to(device)
            
            # 应用内存优化
            print("正在优化内存使用...")
            self.sd_pipeline.enable_attention_slicing()
            
            if hasattr(self.sd_pipeline, 'enable_vae_slicing'):
                self.sd_pipeline.enable_vae_slicing()
                
            if hasattr(self.sd_pipeline, 'enable_vae_tiling'):
                self.sd_pipeline.enable_vae_tiling()
                
            print("模型加载完成！")
            logger.info(f"Successfully initialized SD-Turbo pipeline on {device}")
                
        except Exception as e:
            logger.error(f"Failed to load SD-Turbo model: {str(e)}")
            print(f"模型加载失败: {str(e)}")
            self.sd_pipeline = None
            raise
    
    def _generate_local_image(self, prompt, city_name, aqi_value):
        """
        Generate a simple image locally based on the prompt (fallback method).
        
        Args:
            prompt (str): The prompt for image generation
            city_name (str): Name of the city
            aqi_value (int): AQI value
            
        Returns:
            bytes: The generated image data
        """
        logger.info(f"Generating simple local image for {city_name} with AQI {aqi_value}")
        
        # Get AQI category for color selection
        aqi_category = get_aqi_category(aqi_value)
        
        # Define colors based on AQI category
        colors = {
            "Good": (0, 153, 102),  # Green
            "Moderate": (255, 222, 51),  # Yellow
            "Unhealthy for Sensitive Groups": (255, 153, 51),  # Orange
            "Unhealthy": (204, 0, 51),  # Red
            "Very Unhealthy": (102, 0, 153),  # Purple
            "Hazardous": (126, 0, 35)  # Maroon
        }
        
        # Get the background color based on AQI category
        bg_color = colors.get(aqi_category, (73, 109, 137))  # Default blue if not found
        
        # Create a gradient background
        width, height = 1024, 768
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Add city name
        try:
            title_font = ImageFont.truetype("Arial.ttf", 60)
        except IOError:
            title_font = ImageFont.load_default()
            
        # Center the city name
        city_text = f"{city_name}"
        text_bbox = draw.textbbox((0, 0), city_text, font=title_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 3
        
        # Draw the city name with a shadow effect
        draw.text((text_x+2, text_y+2), city_text, fill=(0, 0, 0), font=title_font)
        draw.text((text_x, text_y), city_text, fill=(255, 255, 255), font=title_font)
        
        # Add AQI visualization
        aqi_text = f"AQI: {aqi_value} - {aqi_category}"
        aqi_bbox = draw.textbbox((0, 0), aqi_text, font=title_font)
        aqi_width = aqi_bbox[2] - aqi_bbox[0]
        aqi_x = (width - aqi_width) // 2
        aqi_y = text_y + text_height + 50
        
        # Draw AQI text
        draw.text((aqi_x+2, aqi_y+2), aqi_text, fill=(0, 0, 0), font=title_font)
        draw.text((aqi_x, aqi_y), aqi_text, fill=(255, 255, 255), font=title_font)
        
        # Add a decorative element based on the prompt
        theme_text = prompt.split("that captures its unique characteristics")[0].split("of ")[-1]
        theme_text = f"Theme: {theme_text}"
        theme_bbox = draw.textbbox((0, 0), theme_text, font=title_font)
        theme_width = theme_bbox[2] - theme_bbox[0]
        theme_x = (width - theme_width) // 2
        theme_y = aqi_y + text_height + 50
        
        # Draw theme text
        draw.text((theme_x+2, theme_y+2), theme_text, fill=(0, 0, 0), font=title_font)
        draw.text((theme_x, theme_y), theme_text, fill=(255, 255, 255), font=title_font)
        
        # Convert to bytes
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
    
    def _generate_local_sd_image(self, prompt, city_name, aqi_value):
        """
        Generate an image using Hugging Face Diffusers with local Stable Diffusion model.
        
        Args:
            prompt (str): The prompt for image generation
            city_name (str): Name of the city
            aqi_value (int): AQI value
            
        Returns:
            bytes: The generated image data
        """
        logger.info(f"Generating image using SD-Turbo for {city_name} with AQI {aqi_value}")
        
        if self.sd_pipeline is None:
            logger.warning("Stable Diffusion pipeline not initialized, falling back to simple image generation")
            return self._generate_local_image(prompt, city_name, aqi_value)
        
        try:
            # Get AQI category for negative prompt enhancement
            aqi_category = get_aqi_category(aqi_value)
            
            # 城市特定的标志性建筑提示 - 简化版本，减少token数量
            landmark_prompts = {
                "北京": "Forbidden City, Great Wall",
                "上海": "Oriental Pearl Tower, Shanghai skyline",
                "广州": "Canton Tower",
                "深圳": "Ping An Finance Center",
                "香港": "Victoria Harbour skyline",
                "台北": "Taipei 101",
                "东京": "Tokyo Tower",
                "大阪": "Osaka Castle",
                "京都": "Kinkaku-ji Temple",
                "首尔": "N Seoul Tower",
                "新加坡": "Marina Bay Sands",
                "曼谷": "Grand Palace",
                "吉隆坡": "Petronas Towers",
                "纽约": "Empire State Building",
                "伦敦": "Big Ben",
                "巴黎": "Eiffel Tower",
                "罗马": "Colosseum",
                "悉尼": "Sydney Opera House",
                "柏林": "Brandenburg Gate",
                "莫斯科": "Saint Basil Cathedral",
                "迪拜": "Burj Khalifa",
                "开罗": "Great Pyramids",
                "里约热内卢": "Christ the Redeemer"
            }
            
            # 获取城市特定的标志性建筑提示，如果没有预定义则使用通用提示
            city_landmark = landmark_prompts.get(city_name, "famous landmark")
            
            # 大幅简化提示词，减少token数量
            if aqi_category in ["Good", "Moderate"]:
                aqi_condition = "clear sky"
                negative_prompt = "smog, haze, pollution, no landmarks, text, words, letters, numbers, watermark"
            elif aqi_category == "Unhealthy for Sensitive Groups":
                aqi_condition = "slight haze"
                negative_prompt = "heavy pollution, no landmarks, text, words, letters, numbers, watermark"
            elif aqi_category == "Unhealthy":
                aqi_condition = "hazy"
                negative_prompt = "no landmarks, text, words, letters, numbers, watermark"
            else:  # Very Unhealthy or Hazardous
                aqi_condition = "smoggy"
                negative_prompt = "clear sky, no landmarks, text, words, letters, numbers, watermark"
            
            # 极度简化提示词，确保不超过77个token
            # 格式：城市地标 + 简短主题 + 天气条件
            enhanced_prompt = f"photo of {city_landmark} in {city_name}, {aqi_condition}, cityscape, no text"
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
            
            # Get hardware info for adaptive settings
            hw_info = get_mac_hardware_info()
            memory_gb = hw_info["memory_gb"]
            is_apple_silicon = hw_info["is_apple_silicon"]
            
            # 根据设备调整参数
            if CUDA_AVAILABLE:
                device = "cuda"
                num_inference_steps = 4
            elif MPS_AVAILABLE and is_apple_silicon:
                device = "mps"
                num_inference_steps = 3
            else:
                device = "cpu"
                num_inference_steps = 2
            
            # 使用合适的分辨率
            if is_apple_silicon and MPS_AVAILABLE:
                width, height = 512, 512
                logger.info("Using 512x512 resolution for Apple Silicon")
            else:
                width, height = 768, 512
            
            # 设置适当的引导系数
            guidance_scale = 7.5
            
            # 简化处理逻辑，避免MPS设备上的问题
            try:
                # 清理GPU缓存
                if device == "cuda":
                    torch.cuda.empty_cache()
                elif device == "mps":
                    torch.mps.empty_cache()
                
                # 对于MPS设备，移动到CPU处理更稳定
                if device == "mps":
                    logger.info("Using CPU for more stable processing on Apple Silicon")
                    self.sd_pipeline.to("cpu")
                    device = "cpu"
                
                # 使用安全模式生成图像
                with torch.inference_mode():
                    # 生成图像
                    image = self.sd_pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height
                    ).images[0]
                    
                # 清理缓存
                if CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error during image generation: {str(e)}")
                
                # 简化重试逻辑
                logger.info("Retrying with CPU and simpler settings...")
                try:
                    # 移动到CPU
                    self.sd_pipeline.to("cpu")
                    
                    # 使用CPU生成，进一步简化提示词
                    simple_prompt = f"photo of {city_landmark}, cityscape, no text"
                    simple_negative = "text, watermark, words, letters, numbers"
                    
                    image = self.sd_pipeline(
                        prompt=simple_prompt,
                        negative_prompt=simple_negative,
                        num_inference_steps=2,  # CPU使用较少步数
                        guidance_scale=7.5,
                        width=512,
                        height=384  # 降低分辨率
                    ).images[0]
                    
                except Exception as inner_e:
                    logger.error(f"Second attempt failed: {str(inner_e)}")
                    logger.warning("Falling back to simple image generation")
                    return self._generate_local_image(prompt, city_name, aqi_value)
            
            # Convert PIL image to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            logger.info(f"Successfully generated image using SD-Turbo for {city_name}")
            return img_byte_arr.getvalue()
            
        except Exception as e:
            logger.error(f"Error in SD-Turbo image generation: {str(e)}")
            logger.warning("Falling back to simple image generation")
            return self._generate_local_image(prompt, city_name, aqi_value)
    
    def _call_bedrock_model(self, prompt):
        """
        Call Amazon Bedrock model to generate an image based on the prompt.
        
        Args:
            prompt (str): The prompt for image generation
            
        Returns:
            bytes: The generated image data
        """
        if self.actual_mode != "bedrock":
            logger.error("Bedrock client not available, but _call_bedrock_model was called")
            raise RuntimeError("Bedrock client not available")
            
        try:
            # Prepare request body based on the model
            if "stability" in self.model_id:
                request_body = {
                    "text_prompts": [{"text": prompt}],
                    "cfg_scale": 8,
                    "steps": 50,
                    "seed": 0,
                    "width": 1024,
                    "height": 768
                }
            else:
                # Default to Titan Image Generator format
                request_body = {
                    "taskType": "TEXT_IMAGE",
                    "textToImageParams": {
                        "text": prompt,
                        "negativeText": "poor quality, blurry, distorted, unrealistic, low resolution",
                    },
                    "imageGenerationConfig": {
                        "numberOfImages": 1,
                        "height": 768,
                        "width": 1024,
                        "cfgScale": 8
                    }
                }
            
            # Make the API call
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse the response based on the model
            response_body = json.loads(response.get('body').read())
            
            if "stability" in self.model_id:
                image_data = base64.b64decode(response_body['artifacts'][0]['base64'])
            else:
                # Default to Titan Image Generator format
                image_data = base64.b64decode(response_body['images'][0])
                
            return image_data
            
        except ClientError as e:
            logger.error(f"Error calling Bedrock: {str(e)}")
            raise
    
    def _add_aqi_overlay(self, image_data, city_name, aqi_value, theme_of_day):
        """
        Add AQI information and health advice overlay to the generated image.
        
        Args:
            image_data (bytes): The raw image data
            city_name (str): Name of the city
            aqi_value (int): AQI value
            theme_of_day (str): Theme of the day
            health_advice (str): Health advice text
            
        Returns:
            BytesIO: The processed image with overlay
        """
        try:
            # Open the image
            image = Image.open(BytesIO(image_data))
            draw = ImageDraw.Draw(image)
            
            # Add semi-transparent overlay at the bottom
            overlay_height = 150
            overlay = Image.new('RGBA', (image.width, overlay_height), (0, 0, 0, 180))
            image.paste(overlay, (0, image.height - overlay_height), overlay)
            
            # 尝试加载字体，如果失败则使用默认字体
            title_font = None
            info_font = None
            
            # 尝试不同的常见字体，适应不同操作系统
            common_fonts = [
                "Arial.ttf", 
                "DejaVuSans.ttf", 
                "FreeSans.ttf", 
                "NotoSans-Regular.ttf",
                "Helvetica.ttf",
                "PingFang.ttc",  # macOS中文字体
                "SimHei.ttf",    # Windows中文字体
                "Microsoft YaHei.ttf"  # Windows中文字体
            ]
            
            # 尝试系统字体目录
            font_dirs = [
                "",  # 当前目录
                "/Library/Fonts/",  # macOS
                "/System/Library/Fonts/",  # macOS
                "C:\\Windows\\Fonts\\",  # Windows
                "/usr/share/fonts/truetype/",  # Linux
                "/usr/share/fonts/",  # Linux
            ]
            
            # 尝试加载字体
            for font_dir in font_dirs:
                for font_name in common_fonts:
                    try:
                        font_path = os.path.join(font_dir, font_name)
                        if os.path.exists(font_path):
                            title_font = ImageFont.truetype(font_path, 36)
                            info_font = ImageFont.truetype(font_path, 24)
                            logger.info(f"Successfully loaded font: {font_path}")
                            break
                    except Exception as e:
                        continue
                if title_font:
                    break
            
            # 如果无法加载任何字体，使用默认字体
            if not title_font:
                logger.warning("Could not load any TrueType fonts, using default font")
                title_font = ImageFont.load_default()
                info_font = ImageFont.load_default()
            
            # 确保城市名称和主题不为空
            if not city_name or city_name == "None":
                city_name = "Unknown City"
            
            if not theme_of_day or theme_of_day == "None":
                theme_of_day = "General"
                
            # 确保主题是字符串
            theme_of_day = self._generate_theme(theme_of_day, aqi_value)
            
            # 获取AQI分类
            aqi_category = get_aqi_category(aqi_value)
            
            # 添加城市名称和主题
            draw.text((20, image.height - overlay_height + 20), 
                     f"{city_name} - {theme_of_day}", fill=(255, 255, 255), font=title_font)
            
            # AQI信息
            draw.text((20, image.height - overlay_height + 80), 
                     f"AQI: {aqi_value} - {aqi_category}", fill=(255, 255, 255), font=info_font)
            
            
            # 转换回字节流
            result = BytesIO()
            image.save(result, format='JPEG')
            result.seek(0)
            return result
            
        except Exception as e:
            logger.error(f"Error adding overlay: {str(e)}")
            # 如果出错，返回原始图像
            result = BytesIO(image_data)
            result.seek(0)
            return result
    
    def _save_to_s3(self, image_data, city_name, theme_of_day):
        """
        Save the generated image to S3 bucket.
        
        Args:
            image_data (BytesIO): The image data
            city_name (str): Name of the city
            theme_of_day (str): Theme of the day
            
        Returns:
            str: The S3 URL of the saved image
        """
        if self.actual_mode == "local":
            logger.info("Local mode: S3 upload simulated")
            return "mock-s3-url-for-testing.jpg"
            
        if not self.s3_bucket:
            logger.warning("No S3 bucket specified, image not saved to S3")
            return None
            
        try:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            safe_city_name = city_name.lower().replace(" ", "_")
            safe_theme = theme_of_day.lower().replace(" ", "_")
            filename = f"{safe_city_name}/{safe_theme}_{timestamp}.jpg"
            
            # Upload to S3
            self.s3_client.upload_fileobj(
                image_data,
                self.s3_bucket,
                f"city_posters/{filename}",
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            
            # Generate the S3 URL
            s3_url = f"https://{self.s3_bucket}.s3.amazonaws.com/city_posters/{filename}"
            return s3_url
            
        except ClientError as e:
            logger.error(f"Error saving to S3: {str(e)}")
            return None
    
    def _generate_theme(self, theme_of_day, aqi_value):
        # 使用datetime直接解析日期字符串，更简单可靠
        from datetime import datetime
        month = None
        try:
            # 尝试解析日期字符串
            if isinstance(theme_of_day, str) and "-" in theme_of_day:
                date_obj = datetime.strptime(theme_of_day, "%Y-%m-%d")
                month = date_obj.month
            else:
                month = datetime.now().month
        except ValueError:
            # 如果解析失败，使用当前月份
            month = datetime.now().month
        
        # 按季节
        if 3 <= month <= 5:
            season = "春日"
        elif 6 <= month <= 8:
            season = "夏日"
        elif 9 <= month <= 11:
            season = "秋日"
        else:
            season = "冬日"
        
        # 基于日期和AQI生成主题
        if aqi_value <= 50:
            mood = "呼吸自由日"
        elif aqi_value <= 100:
            mood = "散步时光"
        elif aqi_value <= 150:
            mood = "口罩出行日"
        elif aqi_value <= 200:
            mood = "注意呼吸健康"
        elif aqi_value <= 300:
            mood = "健康守护日"
        else:
            mood = "避免外出"
        
        theme_of_day = f"{season}·{mood}"
        return theme_of_day
    
    def generate_poster(self, city_name, theme_of_day, aqi_value, save_to_s3=True, output_path=None):
        """
        Generate a personalized city poster with AQI information and health advice.
        
        Args:
            city_name (str): Name of the city
            theme_of_day (str): Theme of the day
            aqi_value (int): AQI value
            save_to_s3 (bool): Whether to save the image to S3
            output_path (str, optional): Local path to save the image
            
        Returns:
            dict: A dictionary containing the image data and metadata
        """
        try:
            logger.info(f"Generating poster for {city_name} with theme '{theme_of_day}' and AQI {aqi_value}")
            
            # 确保参数有效
            if not city_name:
                city_name = "Unknown City"
            
            if not theme_of_day:
                theme_of_day = "General"
            
            # 确保主题是字符串
            theme_of_day = str(theme_of_day)
            
            # Generate the prompt
            prompt = generate_city_poster_prompt(city_name, theme_of_day, aqi_value)
            logger.info(f"Generated prompt: {prompt[:100]}...")
            
            # Generate the image based on the mode
            if self.actual_mode == "bedrock":
                image_data = self._call_bedrock_model(prompt)
                logger.info("Image generated using AWS Bedrock")
            elif self.actual_mode == "local_sd":
                image_data = self._generate_local_sd_image(prompt, city_name, aqi_value)
                logger.info("Image generated using local Stable Diffusion via Hugging Face")
            else:
                image_data = self._generate_local_image(prompt, city_name, aqi_value)
                logger.info("Image generated using simple local method")
            
            # Get AQI category and health advice
            aqi_category = get_aqi_category(aqi_value)
            health_advice = get_health_advice(aqi_category)
            
            # 记录参数值，用于调试
            logger.info(f"Adding overlay with: city_name='{city_name}', theme_of_day='{theme_of_day}', aqi_value={aqi_value}")
            
            # Add AQI overlay
            processed_image = self._add_aqi_overlay(image_data, city_name, aqi_value, theme_of_day)
            
            # Save to S3 if requested and available
            s3_url = None
            if save_to_s3 and self.s3_client:
                s3_url = self._save_to_s3(processed_image, city_name, theme_of_day)
                processed_image.seek(0)  # Reset the file pointer
            
            # Save locally if output_path is provided
            if output_path:
                # Ensure the directory exists
                directory = os.path.dirname(output_path)
                if directory:  # Check if directory is not empty
                    os.makedirs(directory, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(processed_image.getvalue())
                logger.info(f"Image saved to {output_path}")
            
            # Return the result
            return {
                "city_name": city_name,
                "theme_of_day": theme_of_day,
                "aqi_value": aqi_value,
                "aqi_category": aqi_category,
                "health_advice": health_advice,
                "s3_url": s3_url,
                "local_path": output_path if output_path else None,
                "image_data": base64.b64encode(processed_image.getvalue()).decode('utf-8'),
                "generation_mode": self.actual_mode,
                "model_id": self.local_model_id if self.actual_mode == "local_sd" else self.model_id
            }
            
        except Exception as e:
            logger.error(f"Error generating poster: {str(e)}")
            raise

def generate_city_poster(city_name, theme_of_day, aqi_value, output_path=None, s3_bucket=None, 
                      mode="auto", local_model_url="http://localhost:7860"):
    """
    Generate a personalized city poster with AQI information and health advice.
    
    This is a convenience function that uses the CityPosterGenerator class.
    
    Args:
        city_name (str): Name of the city
        theme_of_day (str): Theme of the day
        aqi_value (int): AQI value
        output_path (str, optional): Local path to save the image
        s3_bucket (str, optional): S3 bucket name for storing generated images
        mode (str, optional): Generation mode - "auto", "local", "local_sd", or "bedrock"
            - "auto": Try Bedrock first, fall back to local_sd if available, then local fallback
            - "local": Use simple local image generation (no ML model)
            - "local_sd": Use locally deployed Stable Diffusion model via Hugging Face
            - "bedrock": Use AWS Bedrock only
        local_model_url (str, optional): URL for the local Stable Diffusion API endpoint
        
    Returns:
        dict: A dictionary containing the image data and metadata
    """
    generator = CityPosterGenerator(s3_bucket=s3_bucket, mode=mode, 
                                   local_model_url=local_model_url)
    return generator.generate_poster(
        city_name=city_name,
        theme_of_day=theme_of_day,
        aqi_value=aqi_value,
        save_to_s3=bool(s3_bucket),
        output_path=output_path
    )

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成城市海报")
    parser.add_argument("--mode", type=str, choices=["local", "local_sd", "bedrock", "auto", "all"],
                        default="all", help="指定要测试的模式: local, local_sd, bedrock, auto, all(默认)")
    parser.add_argument("--city", type=str, default=None, help="城市名称")
    parser.add_argument("--theme", type=str, default=None, help="主题")
    parser.add_argument("--aqi", type=int, default=None, help="空气质量指数")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式，显示更多日志")
    
    args = parser.parse_args()
    
    # 如果启用调试模式，设置日志级别为DEBUG
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        logger.setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    # 设置默认值
    city_themes = {
        "local": ("San Francisco", "Foggy Morning", 75, "local_poster.jpg"),
        "local_sd": ("Tokyo", "Cherry Blossom Season", 45, "sd_poster.jpg"),
        "bedrock": ("Beijing", "Autumn Sunset", 120, "bedrock_poster.jpg"),
        "auto": ("Shanghai", "Modern Cityscape", 90, "auto_poster.jpg")
    }
    
    # 如果指定了自定义参数，则使用自定义参数
    custom_city = args.city
    custom_theme = args.theme
    custom_aqi = args.aqi
    custom_output = args.output
    
    # 打印参数信息
    logger.info(f"命令行参数: mode={args.mode}, city={custom_city}, theme={custom_theme}, aqi={custom_aqi}, output={custom_output}")
    
    # 确定要运行的模式
    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["local", "local_sd", "bedrock", "auto"]
    else:
        modes_to_run = [args.mode]
    
    # 运行指定的模式
    for mode in modes_to_run:
        # 获取默认参数
        default_city, default_theme, default_aqi, default_output = city_themes[mode]
        
        # 使用自定义参数或默认参数
        city = custom_city if custom_city else default_city
        theme = custom_theme if custom_theme else default_theme
        aqi = custom_aqi if custom_aqi is not None else default_aqi
        output = custom_output if custom_output else default_output
        
        print(f"\n=== 测试 {mode.upper()} 模式 ===")
        print(f"城市: {city}, 主题: {theme}, AQI: {aqi}, 输出: {output}")
        
        if mode == "local":
            print("使用简单本地模式生成海报...")
            try:
                local_result = generate_city_poster(
                    city_name=city,
                    theme_of_day=theme,
                    aqi_value=aqi,
                    output_path=output,
                    mode="local"
                )
                
                print(f"为 {local_result['city_name']} 生成的海报")
                print(f"AQI 类别: {local_result['aqi_category']}")
                print(f"健康建议: {local_result['health_advice']}")
                print(f"生成模式: {local_result['generation_mode']}")
                if local_result['local_path']:
                    print(f"保存到: {local_result['local_path']}")
            except Exception as e:
                print(f"本地模式生成失败: {str(e)}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        
        elif mode == "local_sd":
            print("尝试使用 Hugging Face Diffusers 生成海报...")
            try:
                # 使用 Stable Diffusion 模型
                sd_result = generate_city_poster(
                    city_name=city,
                    theme_of_day=theme,
                    aqi_value=aqi,
                    output_path=output,
                    mode="local_sd"
                )
                
                print(f"为 {sd_result['city_name']} 生成的海报")
                print(f"AQI 类别: {sd_result['aqi_category']}")
                print(f"健康建议: {sd_result['health_advice']}")
                print(f"生成模式: {sd_result['generation_mode']}")
                print(f"模型: {sd_result['model_id']}")
                if sd_result['local_path']:
                    print(f"保存到: {sd_result['local_path']}")
            except Exception as e:
                print(f"Hugging Face Diffusers 生成失败: {str(e)}")
                print("请确保已安装所需的包: pip install torch diffusers transformers accelerate huggingface_hub tqdm")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        
        elif mode == "bedrock":
            print("尝试使用 AWS Bedrock 生成海报...")
            try:
                bedrock_result = generate_city_poster(
                    city_name=city,
                    theme_of_day=theme,
                    aqi_value=aqi,
                    output_path=output,
                    mode="bedrock"
                )
                
                print(f"为 {bedrock_result['city_name']} 生成的海报")
                print(f"AQI 类别: {bedrock_result['aqi_category']}")
                print(f"健康建议: {bedrock_result['health_advice']}")
                print(f"生成模式: {bedrock_result['generation_mode']}")
                print(f"模型: {bedrock_result['model_id']}")
                if bedrock_result['local_path']:
                    print(f"保存到: {bedrock_result['local_path']}")
                if bedrock_result['s3_url']:
                    print(f"S3 URL: {bedrock_result['s3_url']}")
            except Exception as e:
                print(f"AWS Bedrock 生成失败: {str(e)}")
                print("请确保已配置 AWS 凭证和区域")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        
        elif mode == "auto":
            print("使用自动模式生成海报 (将使用最佳可用方法)...")
            try:
                auto_result = generate_city_poster(
                    city_name=city,
                    theme_of_day=theme,
                    aqi_value=aqi,
                    output_path=output,
                    mode="auto"
                )
                
                print(f"为 {auto_result['city_name']} 生成的海报")
                print(f"AQI 类别: {auto_result['aqi_category']}")
                print(f"健康建议: {auto_result['health_advice']}")
                print(f"生成模式: {auto_result['generation_mode']} (自动选择)")
                print(f"模型: {auto_result['model_id'] if 'model_id' in auto_result else 'N/A'}")
                if auto_result['local_path']:
                    print(f"保存到: {auto_result['local_path']}")
            except Exception as e:
                print(f"自动模式生成失败: {str(e)}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
    
    # 如果运行了多个模式，显示总结
    if len(modes_to_run) > 1:
        print("\n=== 测试完成 ===")
        print(f"运行了 {len(modes_to_run)} 个模式: {', '.join(modes_to_run)}")
