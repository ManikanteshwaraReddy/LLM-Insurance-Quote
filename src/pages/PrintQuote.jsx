import React from 'react'
import { useEffect } from 'react'
import { useNavigate } from 'react-router'

const PrintQuote = () => {
  const navigate = useNavigate();
  useEffect(() => {
    window.print();
    navigate('/');
  }, [navigate])
  
  return (
    <div>
      This is under developement!!!
    </div>
  )
}

export default PrintQuote
