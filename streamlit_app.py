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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                                        url=product.get('url'),
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
        # Placeholder for Home Depot search implementation
        return {
            'product_name': None,
            'brand': None,
            'model_number': product_number,
            'category': None,
            'retailers': {},
            'description': None,
            'specifications': {},
            'error': "Home Depot search not implemented yet"
        }

    async def search_lowes(self, product_number: str, brand: Optional[str] = None) -> Dict:
        """Search Lowes for product information"""
        # Placeholder for Lowes search implementation
        return {
            'product_name': None,
            'brand': None,
            'model_number': product_number,
            'category': None,
            'retailers': {},
            'description': None,
            'specifications': {},
            'error': "Lowes search not implemented yet"
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
    
    # Create form for product entry
    with st.form("product_entry_form"):
        product_number = st.text_input("Product Number")
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
                        
                        # Show retailer links
                        st.subheader("Retailer Links")
                        for retailer, price_info in search_results.retailers.items():
                            if price_info.url:
                                st.markdown(f"[View on {retailer.title()}]({price_info.url})")

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
