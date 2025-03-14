import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
from urllib.parse import quote_plus, urljoin
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from decimal import Decimal
import json
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
from pyzbar.pyzbar import decode
import easyocr
import io

# Add custom CSS for better video display
st.set_page_config(page_title="Toilet Box Scanner", layout="wide")

# Custom CSS to improve video display
st.markdown("""
    <style>
    .stVideo {
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
    }
    .stVideo > video {
        width: 100%;
        height: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize EasyOCR reader (this will download the model on first run)
if 'ocr_reader' not in st.session_state:
    st.session_state.ocr_reader = easyocr.Reader(['en'])

@dataclass
class ProductPrice:
    price: Optional[Decimal]
    url: str
    in_stock: bool
    model_number: Optional[str]
    sku: Optional[str]
    raw_price: str

@dataclass
class SearchResult:
    product_name: Optional[str]
    brand: Optional[str]
    model_number: Optional[str]
    category: Optional[str]
    retailers: Dict[str, ProductPrice]
    description: Optional[str]
    specifications: Dict[str, str]
    error: Optional[str]

class RetailerError(Exception):
    """Base exception for retailer-specific errors"""
    pass

class ProductSearcher:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def clean_price(self, price_str: str) -> Optional[Decimal]:
        """Clean and convert price string to Decimal"""
        try:
            if not price_str or price_str == 'N/A':
                return None
            # Remove currency symbols and convert to number
            cleaned = re.sub(r'[^\d.]', '', price_str)
            return Decimal(cleaned) if cleaned else None
        except Exception as e:
            self.logger.error(f"Error cleaning price {price_str}: {str(e)}")
            return None

    async def search_all_retailers(self, product_number: str, brand: Optional[str] = None) -> SearchResult:
        """Search all retailers for a product"""
        self.logger.info(f"Searching for product: {product_number} (Brand: {brand})")
        
        # Validate input
        if not self._validate_product_number(product_number):
            return SearchResult(
                product_name=None,
                brand=brand,
                model_number=product_number,
                category=None,
                retailers={},
                description=None,
                specifications={},
                error="Invalid product number format"
            )

        # Search tasks
        tasks = [
            self._safe_search(self.search_ferguson, product_number, brand, "Ferguson"),
            self._safe_search(self.search_homedepot, product_number, brand, "Home Depot"),
            self._safe_search(self.search_lowes, product_number, brand, "Lowes")
        ]

        results = await asyncio.gather(*tasks)
        
        # Combine results
        combined = self._combine_search_results(results, product_number, brand)
        self.logger.info(f"Search completed for {product_number}. Found {len(combined.retailers)} results")
        return combined

    async def _safe_search(self, search_func, product_number: str, brand: Optional[str], retailer_name: str) -> Dict:
        """Wrapper for safe execution of search functions"""
        try:
            return await search_func(product_number, brand)
        except Exception as e:
            self.logger.error(f"Error searching {retailer_name}: {str(e)}")
            return {
                'error': f"Error searching {retailer_name}: {str(e)}",
                'retailers': {}
            }

    async def search_ferguson(self, product_number: str, brand: Optional[str] = None) -> Dict:
        """Search Ferguson for product information"""
        base_url = "https://www.ferguson.com"
        search_url = f"{base_url}/search/{quote_plus(f'{brand or ''} {product_number}')}"
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract product information
                        product = self._extract_ferguson_product(soup)
                        if product:
                            return {
                                'product_name': product.get('name'),
                                'brand': product.get('brand'),
                                'model_number': product_number,
                                'category': product.get('category'),
                                'retailers': {
                                    'ferguson': ProductPrice(
                                        price=self.clean_price(product.get('price')),
                                        url=search_url,
                                        in_stock=product.get('in_stock', False),
                                        model_number=product_number,
                                        sku=product.get('sku'),
                                        raw_price=product.get('price', 'N/A')
                                    )
                                },
                                'description': product.get('description'),
                                'specifications': product.get('specifications', {})
                            }
            except Exception as e:
                self.logger.error(f"Error searching Ferguson: {str(e)}")
        
        return {
            'product_name': None,
            'brand': None,
            'model_number': product_number,
            'category': None,
            'retailers': {},
            'description': None,
            'specifications': {},
            'error': "No results found on Ferguson"
        }

    def _extract_ferguson_product(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract product information from Ferguson page"""
        product = {}
        
        # Try to get structured data first
        structured_data = self._extract_structured_data(soup)
        if structured_data:
            product.update(structured_data)
        
        # Extract basic information
        product_elem = soup.find('div', {'class': 'product-details'})
        if product_elem:
            name_elem = product_elem.find('h1', {'class': 'product-title'})
            if name_elem:
                product['name'] = name_elem.text.strip()
            
            price_elem = product_elem.find('span', {'class': 'price'})
            if price_elem:
                product['price'] = price_elem.text.strip()
            
            brand_elem = product_elem.find('span', {'class': 'brand'})
            if brand_elem:
                product['brand'] = brand_elem.text.strip()
            
            # Check if in stock
            stock_elem = product_elem.find('span', {'class': 'stock-status'})
            if stock_elem:
                product['in_stock'] = 'in stock' in stock_elem.text.lower()
        
        return product

    def _validate_product_number(self, product_number: str) -> bool:
        """Validate product number format"""
        if not product_number:
            return False
        
        # Common patterns for different brands
        patterns = {
            'kohler': r'^[Kk]-\d{4}(-\d+)?$',
            'toto': r'^[Cc][Ss][Tt]\d{3,4}[A-Za-z]?$',
            'american_standard': r'^[0-9A-Z]{4,8}$',
            'delta': r'^[0-9A-Z]{3,10}$',
            'moen': r'^[0-9A-Z]{4,12}$'
        }
        
        return any(re.match(pattern, product_number) for pattern in patterns.values())

    def _extract_structured_data(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract structured data from page"""
        try:
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    return json.loads(script.string)
                except:
                    continue
        except Exception as e:
            self.logger.warning(f"Error extracting structured data: {str(e)}")
            return None

    def _combine_search_results(self, results: list, product_number: str, brand: Optional[str]) -> SearchResult:
        """Combine results from multiple retailers"""
        combined = SearchResult(
            product_name=None,
            brand=brand,
            model_number=product_number,
            category=None,
            retailers={},
            description=None,
            specifications={},
            error=None
        )

        errors = []
        for result in results:
            if result.get('error'):
                errors.append(result['error'])
            
            # Update product info if not already set
            if not combined.product_name and result.get('product_name'):
                combined.product_name = result['product_name']
            if not combined.brand and result.get('brand'):
                combined.brand = result['brand']
            if not combined.category and result.get('category'):
                combined.category = result['category']
            
            # Merge retailers
            if 'retailers' in result:
                combined.retailers.update(result['retailers'])
            
            # Merge specifications
            if 'specifications' in result:
                combined.specifications.update(result['specifications'])

        if errors and not combined.retailers:
            combined.error = "; ".join(errors)

        return combined

    async def search_homedepot(self, product_number: str, brand: Optional[str] = None) -> Dict:
        """Search Home Depot for product information"""
        base_url = "https://www.homedepot.com"
        search_url = f"{base_url}/s/{quote_plus(f'{brand or ''} {product_number}')}"
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        # Add Home Depot specific scraping here
                        return {
                            'product_name': None,
                            'brand': brand,
                            'model_number': product_number,
                            'category': None,
                            'retailers': {
                                'homedepot': ProductPrice(
                                    price=None,
                                    url=search_url,
                                    in_stock=False,
                                    model_number=product_number,
                                    sku=None,
                                    raw_price='Check website'
                                )
                            }
                        }
            except Exception as e:
                self.logger.error(f"Error searching Home Depot: {str(e)}")
        
        return {
            'retailers': {},
            'error': "Home Depot search implementation pending"
        }

    async def search_lowes(self, product_number: str, brand: Optional[str] = None) -> Dict:
        """Search Lowes for product information"""
        base_url = "https://www.lowes.com"
        search_url = f"{base_url}/search?searchTerm={quote_plus(f'{brand or ''} {product_number}')}"
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(search_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        # Add Lowes specific scraping here
                        return {
                            'product_name': None,
                            'brand': brand,
                            'model_number': product_number,
                            'category': None,
                            'retailers': {
                                'lowes': ProductPrice(
                                    price=None,
                                    url=search_url,
                                    in_stock=False,
                                    model_number=product_number,
                                    sku=None,
                                    raw_price='Check website'
                                )
                            }
                        }
            except Exception as e:
                self.logger.error(f"Error searching Lowes: {str(e)}")
        
        return {
            'retailers': {},
            'error': "Lowes search implementation pending"
        }

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_barcode = None
        self.last_ocr_text = None
        self.frame_count = 0
        self.ocr_interval = 15  # Reduced interval for more frequent scanning
        self.last_successful_scan = 0
        self.scan_cooldown = 30  # Frames to wait before showing the same result again

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Enhance image for better detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)  # Increase contrast
        
        # Scan for barcodes with enhanced image
        try:
            barcodes = decode(enhanced)
            for barcode in barcodes:
                # Draw rectangle around barcode
                points = np.array([barcode.polygon], np.int32)
                cv2.polylines(img, [points], True, (0, 255, 0), 2)
                
                # Store the barcode data
                barcode_data = barcode.data.decode('utf-8')
                if barcode_data != self.last_barcode or self.frame_count - self.last_successful_scan > self.scan_cooldown:
                    self.last_barcode = barcode_data
                    self.last_successful_scan = self.frame_count
                
                # Draw the barcode data with better visibility
                cv2.putText(img, self.last_barcode, 
                          (barcode.rect.left, barcode.rect.top - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw a filled rectangle for better text visibility
                text_size = cv2.getTextSize(self.last_barcode, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(img, 
                            (barcode.rect.left - 2, barcode.rect.top - text_size[1] - 15),
                            (barcode.rect.left + text_size[0] + 2, barcode.rect.top - 5),
                            (0, 0, 0), -1)
                cv2.putText(img, self.last_barcode,
                          (barcode.rect.left, barcode.rect.top - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            logger.error(f"Error in barcode detection: {str(e)}")

        # Perform OCR more frequently
        self.frame_count += 1
        if self.frame_count % self.ocr_interval == 0:
            try:
                # Use both original and enhanced images for OCR
                results = st.session_state.ocr_reader.readtext(enhanced)
                
                if results:
                    # Filter results with confidence threshold
                    filtered_results = [r for r in results if r[2] > 0.45]  # Confidence threshold
                    
                    # Combine all detected text
                    detected_text = ' '.join([text[1] for text in filtered_results])
                    
                    # Update only if new text is found or enough frames have passed
                    if (detected_text != self.last_ocr_text and detected_text.strip()) or \
                       (self.frame_count - self.last_successful_scan > self.scan_cooldown):
                        self.last_ocr_text = detected_text
                        self.last_successful_scan = self.frame_count
                    
                    # Draw boxes around detected text with improved visibility
                    for (bbox, text, prob) in filtered_results:
                        # Convert bbox points to integers
                        bbox = np.array(bbox).astype(int)
                        
                        # Draw the bounding box
                        cv2.polylines(img, [bbox], True, (255, 0, 0), 2)
                        
                        # Draw a filled rectangle for better text visibility
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(img, 
                                    (bbox[0][0] - 2, bbox[0][1] - text_size[1] - 15),
                                    (bbox[0][0] + text_size[0] + 2, bbox[0][1] - 5),
                                    (0, 0, 0), -1)
                        
                        # Add text above the box with better visibility
                        cv2.putText(img, f"{text} ({prob:.2f})",
                                  (bbox[0][0], bbox[0][1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            except Exception as e:
                logger.error(f"Error in OCR: {str(e)}")

        # Add scanning guide overlay
        height, width = img.shape[:2]
        guide_color = (255, 255, 255)
        guide_thickness = 2
        
        # Draw scanning area guide
        margin = 50
        cv2.rectangle(img, (margin, margin), (width - margin, height - margin), guide_color, guide_thickness)
        
        # Add helper text
        helper_text = "Center barcode or text in box"
        text_size = cv2.getTextSize(helper_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(img, helper_text, (text_x, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, guide_color, 2)

        return img

    def get_scan_data(self):
        return {
            'barcode': self.last_barcode,
            'ocr_text': self.last_ocr_text
        }
# Initialize session state for inventory data
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = pd.DataFrame(columns=[
        'Date Added', 'Product Number', 'Brand', 'Model Name', 'Category',
        'Quantity', 'MSRP', 'Notes'
    ])

def main():
    st.title("Toilet Box Scanner Inventory Management")
    
    # Initialize ProductSearcher
    searcher = ProductSearcher()
    
    # Add tabs for different input methods
    tab1, tab2 = st.tabs(["📷 Scan Product", "✍️ Manual Entry"])
    
    with tab1:
        st.header("Scan Product")
        st.write("Point your camera at the product barcode or text to scan.")
        
        # Initialize the webcam scanner with specific configurations
        ctx = webrtc_streamer(
            key="product-scanner",
            video_transformer_factory=VideoTransformer,
            async_transform=True,
            media_stream_constraints={
                "video": {
                    "width": 640,
                    "height": 480,
                    "facingMode": "environment"  # Use back camera if available
                },
                "audio": False  # Disable audio
            },
            rtc_configuration={  # Add STUN/TURN servers for better connectivity
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },
            video_html_attrs={
                "style": {"width": "100%", "height": "100%"},
                "controls": False,
            }
        )
        
        if ctx.video_transformer:
            scan_data = ctx.video_transformer.get_scan_data()
            
            if scan_data['barcode'] or scan_data['ocr_text']:
                st.success("Product detected!")
                
                if scan_data['barcode']:
                    st.write("📊 Barcode:", scan_data['barcode'])
                    
                if scan_data['ocr_text']:
                    # Try to extract product number using regex patterns
                    patterns = {
                        'kohler': r'[Kk]-\d{4}(-\d+)?',
                        'toto': r'[Cc][Ss][Tt]\d{3,4}[A-Za-z]?',
                        'american_standard': r'[0-9A-Z]{4,8}',
                    }
                    
                    found_product_number = None
                    for pattern in patterns.values():
                        match = re.search(pattern, scan_data['ocr_text'])
                        if match:
                            found_product_number = match.group(0)
                            break
                    
                    if found_product_number:
                        st.write("📝 Product Number:", found_product_number)
                        
                        # Add button to use scanned product number
                        if st.button("Use This Product Number"):
                            st.session_state['scanned_product'] = found_product_number
                            st.experimental_rerun()
        else:
            st.warning("Camera not available. Please check your camera permissions and try again.")
            # Add troubleshooting tips
            with st.expander("Troubleshooting Tips"):
                st.markdown("""
                If you can't see your camera feed:
                1. Make sure you've allowed camera access in your browser
                2. Try refreshing the page
                3. Check if another application is using your camera
                4. Try using a different browser (Chrome or Firefox recommended)
                """)
    with tab2:
        st.header("Manual Product Entry")
        
        # Create form for product entry
        with st.form("product_entry_form"):
            # Pre-fill product number if scanned
            initial_product_number = st.session_state.get('scanned_product', '')
            product_number = st.text_input("Product Number", value=initial_product_number)
            brand_options = ["Kohler", "Toto", "American Standard", "Delta", "Moen", "Other"]
            brand = st.selectbox("Brand", brand_options)
            model_name = st.text_input("Model Name")
            category = st.text_input("Category")
            quantity = st.number_input("Quantity", min_value=1, value=1)
            msrp = st.number_input("MSRP", min_value=0.0, value=0.0)
            notes = st.text_area("Notes")
            
            submit_button = st.form_submit_button("Add Product")
            
            if submit_button:
                try:
                    # Validate product number
                    if not product_number:
                        st.error("Please enter a product number.")
                        return
                    
                    # Add product with error handling
                    with st.spinner('Searching retailers for pricing...'):
                        try:
                            search_results = asyncio.run(searcher.search_all_retailers(
                                product_number=product_number,
                                brand=brand if brand != "Other" else None
                            ))
                            
                            if search_results.error:
                                st.warning(f"Some searches failed: {search_results.error}")
                            
                            # Update form with found information
                            if search_results.brand and brand == "Other":
                                brand = search_results.brand
                            if not model_name and search_results.product_name:
                                model_name = search_results.product_name
                            if not category and search_results.category:
                                category = search_results.category

                            # Create new row with enhanced information
                            new_row = {
                                'Date Added': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'Product Number': product_number,
                                'Brand': brand,
                                'Model Name': model_name,
                                'Category': category,
                                'Quantity': quantity,
                                'MSRP': msrp,
                                'Notes': notes
                            }

                            # Add retailer information
                            for retailer, price_info in search_results.retailers.items():
                                new_row[f'{retailer.title()} Price'] = price_info.raw_price
                                new_row[f'{retailer.title()} Link'] = price_info.url
                                new_row[f'{retailer.title()} In Stock'] = 'Yes' if price_info.in_stock else 'No'

                            # Add to inventory
                            st.session_state.inventory_data = pd.concat([
                                st.session_state.inventory_data,
                                pd.DataFrame([new_row])
                            ], ignore_index=True)

                            # Show success message with details
                            st.success("Product added to inventory!")
                            
                            # Show price comparison and retailer links in a table
                            if search_results.retailers:
                                st.subheader("Price Comparison")
                                
                                # Create a price comparison table
                                comparison_data = []
                                for retailer, price_info in search_results.retailers.items():
                                    comparison_data.append({
                                        "Retailer": retailer.title(),
                                        "Price": price_info.raw_price,
                                        "In Stock": "Yes" if price_info.in_stock else "No",
                                        "Link": price_info.url
                                    })
                                
                                if comparison_data:
                                    comparison_df = pd.DataFrame(comparison_data)
                                    
                                    # Display the comparison table
                                    st.dataframe(
                                        comparison_df,
                                        column_config={
                                            "Link": st.column_config.LinkColumn("Store Link"),
                                            "Price": st.column_config.TextColumn("Price", help="Current price at retailer"),
                                            "In Stock": st.column_config.TextColumn("Availability")
                                        },
                                        hide_index=True
                                    )
                                    
                                    # Show the best price
                                    try:
                                        best_price = min(
                                            (price_info for price_info in search_results.retailers.values() if price_info.price is not None),
                                            key=lambda x: x.price,
                                            default=None
                                        )
                                        if best_price:
                                            st.info(f"💰 Best price found: ${best_price.price} at {best_price.url}")
                                    except Exception as e:
                                        logger.error(f"Error calculating best price: {str(e)}")

                            # Show specifications if available
                            if search_results.specifications:
                                with st.expander("Product Specifications"):
                                    for key, value in search_results.specifications.items():
                                        st.text(f"{key}: {value}")

                        except Exception as e:
                            st.error(f"Error adding product: {str(e)}")
                            logger.error(f"Error adding product {product_number}: {str(e)}", exc_info=True)

                except Exception as e:
                    st.error(f"Error processing form: {str(e)}")
                    logger.error(f"Error processing form: {str(e)}", exc_info=True)

    # Display inventory table
    if not st.session_state.inventory_data.empty:
        st.subheader("Current Inventory")
        st.dataframe(st.session_state.inventory_data)

if __name__ == "__main__":
    main()
