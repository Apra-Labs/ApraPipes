#pragma once
#include <boost/container/deque.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/call_traits.hpp>

template <class T>
class threadsafe_que
{
public:

	typedef boost::container::deque<T> container_type;
	typedef typename container_type::size_type size_type;
	typedef typename container_type::value_type value_type;
	
	explicit threadsafe_que() : m_unread(0), m_wakeExternally(false) {}

	void push(typename boost::call_traits<value_type>::param_type item)
	{ // `param_type` represents the "best" way to pass a parameter of type `value_type` to a method.

		boost::mutex::scoped_lock lock(m_mutex);
		m_container.push_front(item);
		++m_unread;				
		lock.unlock();
		m_not_empty.notify_one();
	}

	value_type try_pop() {
		boost::mutex::scoped_lock lock(m_mutex);
		if (is_not_empty())
		{
			--m_unread;
			value_type ret = m_container.back();
			m_container.pop_back();
			lock.unlock();
			return ret;
		}
		else {
			lock.unlock();
			return value_type();//empty container
		}
	}

	value_type pop() {
		boost::mutex::scoped_lock lock(m_mutex);
		m_not_empty.wait(lock, boost::bind(&threadsafe_que<value_type>::is_not_empty, this));
		--m_unread;
		value_type ret = m_container.back();
		m_container.pop_back();
		lock.unlock();
		return ret;
	}

	void clear() {
		boost::mutex::scoped_lock lock(m_mutex);
		m_container.clear();
		m_unread = 0;
		lock.unlock();
	}	

	size_t size()
	{
		boost::mutex::scoped_lock lock(m_mutex);
		return m_unread;
	}

	value_type try_pop_external() {
		boost::mutex::scoped_lock lock(m_mutex);
		m_not_empty.wait(lock, boost::bind(&threadsafe_que<value_type>::is_not_empty_external, this));
		if(!is_not_empty()) {
			lock.unlock();
			return value_type();//empty container
		}
		--m_unread;
		value_type ret = m_container.back();
		m_container.pop_back();
		lock.unlock();
		return ret;
	}

	void setWake()
	{
		boost::mutex::scoped_lock lock(m_mutex);
		m_wakeExternally = true; // forever true - now try_pop_external becomes equivalent to try_pop
		m_not_empty.notify_one();
	}

private:
	threadsafe_que(const threadsafe_que&);              // Disabled copy constructor.
	threadsafe_que& operator = (const threadsafe_que&); // Disabled assign operator.

	bool is_not_empty() const { return m_unread > 0; }

	bool is_not_empty_external() const 
	{ 
		return m_unread > 0 || m_wakeExternally; 
	}	

	bool m_wakeExternally;
	size_type m_unread;
	container_type m_container;
	boost::mutex m_mutex;	
	boost::condition m_not_empty;
};
