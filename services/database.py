# services/database.py
# Complete database service with menu reading for AI

import os
import asyncpg
from typing import Dict, List, Optional, Any
from datetime import datetime
from services.log_utils import Log

class DatabaseService:
    """Database service for managing restaurant data and AI queries."""
    
    def __init__(self):
        self.pool = None
        self.database_url = os.getenv("DATABASE_URL")
        self._menu_cache = {}  # Cache menu in memory
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update = {}
        
    async def connect(self):
        """Initialize database connection pool."""
        try:
            if not self.database_url:
                Log.warning("⚠️ DATABASE_URL not set - database features disabled")
                return False
                
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            Log.info("✅ Database connected successfully")
            return True
        except Exception as e:
            Log.error(f"❌ Database connection failed: {e}")
            return False
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            Log.info("🔌 Database connection closed")
    
    # ==================== RESTAURANT INFO ====================
    
    async def get_restaurant_info(self, restaurant_id: str) -> Optional[Dict]:
        """Get restaurant information for AI context."""
        try:
            if not self.pool:
                return None
                
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT 
                        name, phone, address, cuisine_type,
                        delivery_fee, minimum_order, estimated_delivery_time,
                        payment_methods, opening_hours, special_instructions
                    FROM restaurants
                    WHERE restaurant_id = $1 AND is_active = true
                    """,
                    restaurant_id
                )
                
                if not row:
                    Log.warning(f"⚠️ Restaurant not found: {restaurant_id}")
                    return None
                
                info = dict(row)
                Log.info(f"📋 Loaded restaurant info: {info['name']}")
                return info
                
        except Exception as e:
            Log.error(f"❌ Failed to get restaurant info: {e}")
            return None
    
    # ==================== MENU QUERIES FOR AI ====================
    
    async def get_full_menu(self, restaurant_id: str) -> Dict[str, Any]:
        """
        Get complete menu for AI to reference.
        Returns structured menu with categories and items.
        """
        try:
            if not self.pool:
                return {"categories": [], "items": []}
            
            # Check cache
            cache_key = f"menu_{restaurant_id}"
            now = datetime.now().timestamp()
            
            if cache_key in self._menu_cache:
                last_update = self._last_cache_update.get(cache_key, 0)
                if now - last_update < self._cache_ttl:
                    Log.debug(f"📦 Using cached menu for {restaurant_id}")
                    return self._menu_cache[cache_key]
            
            async with self.pool.acquire() as conn:
                # Get restaurant UUID
                restaurant = await conn.fetchrow(
                    "SELECT id FROM restaurants WHERE restaurant_id = $1",
                    restaurant_id
                )
                
                if not restaurant:
                    return {"categories": [], "items": []}
                
                restaurant_uuid = restaurant['id']
                
                # Get categories
                categories = await conn.fetch(
                    """
                    SELECT id, name, description
                    FROM menu_categories
                    WHERE restaurant_id = $1 AND is_active = true
                    ORDER BY display_order, name
                    """,
                    restaurant_uuid
                )
                
                # Get all menu items
                items = await conn.fetch(
                    """
                    SELECT 
                        mi.id, mi.name, mi.description, mi.price,
                        mi.is_available, mi.is_popular, mi.is_spicy, 
                        mi.is_vegetarian, mi.allergens,
                        mc.name as category_name
                    FROM menu_items mi
                    LEFT JOIN menu_categories mc ON mi.category_id = mc.id
                    WHERE mi.restaurant_id = $1
                    ORDER BY mc.display_order, mi.display_order, mi.name
                    """,
                    restaurant_uuid
                )
                
                menu = {
                    "categories": [dict(c) for c in categories],
                    "items": [dict(i) for i in items]
                }
                
                # Cache it
                self._menu_cache[cache_key] = menu
                self._last_cache_update[cache_key] = now
                
                Log.info(f"📋 Loaded menu: {len(menu['items'])} items in {len(menu['categories'])} categories")
                return menu
                
        except Exception as e:
            Log.error(f"❌ Failed to get menu: {e}")
            return {"categories": [], "items": []}
    
    async def search_menu_items(self, restaurant_id: str, query: str) -> List[Dict]:
        """Search menu items by name or description."""
        try:
            if not self.pool:
                return []
                
            async with self.pool.acquire() as conn:
                restaurant = await conn.fetchrow(
                    "SELECT id FROM restaurants WHERE restaurant_id = $1",
                    restaurant_id
                )
                
                if not restaurant:
                    return []
                
                items = await conn.fetch(
                    """
                    SELECT 
                        name, description, price, is_available,
                        is_popular, is_spicy, is_vegetarian
                    FROM menu_items
                    WHERE restaurant_id = $1 
                    AND (
                        name ILIKE $2 
                        OR description ILIKE $2
                    )
                    AND is_available = true
                    ORDER BY is_popular DESC, name
                    LIMIT 10
                    """,
                    restaurant['id'],
                    f"%{query}%"
                )
                
                return [dict(item) for item in items]
                
        except Exception as e:
            Log.error(f"❌ Menu search failed: {e}")
            return []
    
    async def get_item_price(self, restaurant_id: str, item_name: str) -> Optional[float]:
        """Get price of a specific menu item."""
        try:
            if not self.pool:
                return None
                
            async with self.pool.acquire() as conn:
                restaurant = await conn.fetchrow(
                    "SELECT id FROM restaurants WHERE restaurant_id = $1",
                    restaurant_id
                )
                
                if not restaurant:
                    return None
                
                item = await conn.fetchrow(
                    """
                    SELECT price
                    FROM menu_items
                    WHERE restaurant_id = $1 
                    AND name ILIKE $2
                    AND is_available = true
                    LIMIT 1
                    """,
                    restaurant['id'],
                    item_name
                )
                
                if item:
                    return float(item['price'])
                return None
                
        except Exception as e:
            Log.error(f"❌ Failed to get item price: {e}")
            return None
    
    # ==================== CALL MANAGEMENT ====================
    
    async def create_call(
        self,
        call_sid: str,
        restaurant_id: str,
        caller_phone: Optional[str] = None
    ) -> Optional[str]:
        """Create a new call record. Returns UUID."""
        try:
            if not self.pool:
                return None
                
            async with self.pool.acquire() as conn:
                # Get restaurant UUID
                restaurant = await conn.fetchrow(
                    "SELECT id FROM restaurants WHERE restaurant_id = $1",
                    restaurant_id
                )
                
                if not restaurant:
                    Log.warning(f"⚠️ Restaurant not found: {restaurant_id}")
                    return None
                
                row = await conn.fetchrow(
                    """
                    INSERT INTO calls (call_sid, restaurant_id, caller_phone, start_time)
                    VALUES ($1, $2, $3, NOW())
                    RETURNING id
                    """,
                    call_sid, restaurant['id'], caller_phone
                )
                
                call_uuid = str(row['id'])
                Log.info(f"📞 Created call record: {call_uuid}")
                return call_uuid
                
        except Exception as e:
            Log.error(f"❌ Failed to create call: {e}")
            return None
    
    async def end_call(self, call_sid: str, duration_seconds: int):
        """Mark call as ended."""
        try:
            if not self.pool:
                return
                
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE calls
                    SET end_time = NOW(),
                        duration_seconds = $1,
                        status = 'completed',
                        updated_at = NOW()
                    WHERE call_sid = $2
                    """,
                    duration_seconds, call_sid
                )
                Log.info(f"📞 Ended call: {call_sid}")
                
        except Exception as e:
            Log.error(f"❌ Failed to end call: {e}")
    
    # ==================== TRANSCRIPT MANAGEMENT ====================
    
    async def add_transcript(
        self,
        call_sid: str,
        speaker: str,
        text: str,
        timestamp: datetime,
        confidence: Optional[float] = None
    ):
        """Add transcript entry."""
        try:
            if not self.pool:
                return
                
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO transcripts (call_id, speaker, text, timestamp, confidence)
                    SELECT id, $2, $3, $4, $5
                    FROM calls
                    WHERE call_sid = $1
                    """,
                    call_sid, speaker, text, timestamp, confidence
                )
                
        except Exception as e:
            Log.error(f"❌ Failed to add transcript: {e}")
    
    # ==================== ORDER MANAGEMENT ====================
    
    async def create_order(
        self,
        call_sid: str,
        order_data: Dict[str, Any]
    ) -> Optional[str]:
        """Create order from call data. Returns order UUID."""
        try:
            if not self.pool:
                return None
                
            async with self.pool.acquire() as conn:
                # Get call info
                call = await conn.fetchrow(
                    "SELECT id, restaurant_id FROM calls WHERE call_sid = $1",
                    call_sid
                )
                
                if not call:
                    Log.warning(f"⚠️ Call not found: {call_sid}")
                    return None
                
                # Create order
                order_row = await conn.fetchrow(
                    """
                    INSERT INTO orders (
                        call_id, restaurant_id, customer_name, customer_phone,
                        delivery_address, subtotal, delivery_fee, tax, total_price,
                        payment_method, delivery_instructions, estimated_delivery_time
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    RETURNING id
                    """,
                    call['id'],
                    call['restaurant_id'],
                    order_data.get('customer_name'),
                    order_data.get('phone_number'),
                    order_data.get('address'),
                    order_data.get('subtotal', 0),
                    order_data.get('delivery_fee', 0),
                    order_data.get('tax', 0),
                    order_data.get('total_price', 0),
                    order_data.get('payment_method'),
                    order_data.get('delivery_instructions'),
                    order_data.get('estimated_delivery_time')
                )
                
                order_uuid = order_row['id']
                
                # Add order items
                items = order_data.get('order_items', [])
                for item in items:
                    await conn.execute(
                        """
                        INSERT INTO order_items (
                            order_id, item_name, quantity, unit_price, 
                            total_price, special_instructions
                        )
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        order_uuid,
                        item.get('name'),
                        item.get('quantity', 1),
                        item.get('unit_price', 0),
                        item.get('total_price', 0),
                        item.get('special_instructions')
                    )
                
                Log.info(f"📦 Created order: {order_uuid} with {len(items)} items")
                return str(order_uuid)
                
        except Exception as e:
            Log.error(f"❌ Failed to create order: {e}")
            return None
    
    # ==================== ANALYTICS ====================
    
    async def get_daily_stats(self, restaurant_id: str) -> Dict:
        """Get today's statistics."""
        try:
            if not self.pool:
                return {}
                
            async with self.pool.acquire() as conn:
                restaurant = await conn.fetchrow(
                    "SELECT id FROM restaurants WHERE restaurant_id = $1",
                    restaurant_id
                )
                
                if not restaurant:
                    return {}
                
                stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(*) as total_calls,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_calls,
                        AVG(duration_seconds) as avg_duration
                    FROM calls
                    WHERE restaurant_id = $1
                    AND DATE(start_time) = CURRENT_DATE
                    """,
                    restaurant['id']
                )
                
                return dict(stats)
                
        except Exception as e:
            Log.error(f"❌ Failed to get stats: {e}")
            return {}

# Global database instance
db = DatabaseService()
