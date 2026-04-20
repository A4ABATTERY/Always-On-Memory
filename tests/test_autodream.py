"""
Simulation tests for the AutoDream background optimization loop.
"""

import unittest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from agent import MemoryAgent, autodream_loop

class TestAutoDream(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        # Patch build_agents to return mocks
        self.mock_agents = [MagicMock() for _ in range(9)]
        for m in self.mock_agents:
            m.run = AsyncMock()
            
        with patch('agent.build_agents', return_value=self.mock_agents):
            with patch('agent.init_db'):
                self.agent = MemoryAgent()

    async def test_autodream_trigger(self):
        """Test that autodream triggers optimization when idle."""
        # Mock dependencies in agent.py
        with patch('agent.system_is_idle', return_value=True):
            with patch('agent._dream_decay', new_callable=AsyncMock) as mock_decay:
                with patch('agent._dream_reorganize', new_callable=AsyncMock) as mock_reorg:
                    # Mock shutdown event to exit immediately after one iteration
                    mock_shutdown = MagicMock()
                    mock_shutdown.is_set.side_effect = [False, True]
                    
                    with patch('agent._shutdown_event', mock_shutdown):
                        # We need to use wait_for because autodream_loop has an await asyncio.wait_for
                        # Let's mock the sleep/wait in the loop too
                        with patch('asyncio.wait_for', side_effect=[asyncio.TimeoutError(), None]):
                            await autodream_loop(self.agent, check_interval=0.1)
                    
                    mock_decay.assert_called_once()
                    mock_reorg.assert_called_once_with(self.agent)

    async def test_autodream_skips_when_busy(self):
        """Test that autodream skips optimization when system is not idle."""
        with patch('agent.system_is_idle', return_value=False):
            with patch('agent._dream_decay', new_callable=AsyncMock) as mock_decay:
                mock_shutdown = MagicMock()
                mock_shutdown.is_set.side_effect = [False, True]
                
                with patch('agent._shutdown_event', mock_shutdown):
                    with patch('asyncio.wait_for', side_effect=[asyncio.TimeoutError(), None]):
                        await autodream_loop(self.agent, check_interval=0.1)
                
                mock_decay.assert_not_called()

if __name__ == "__main__":
    unittest.main()
