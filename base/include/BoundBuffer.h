#pragma once
#include <boost/container/deque.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>
#include <boost/call_traits.hpp>
#include <boost/bind.hpp>

template <class T>
class bounded_buffer
{
public:

	typedef boost::container::deque<T> container_type;
	typedef typename container_type::size_type size_type;
	typedef typename container_type::value_type value_type;
	typedef typename boost::call_traits<value_type>::param_type param_type;

	explicit bounded_buffer(size_type capacity) : m_unread(0), m_capacity(capacity), m_accept(true) {}

	void push(typename boost::call_traits<value_type>::param_type item)
	{ // `param_type` represents the "best" way to pass a parameter of type `value_type` to a method.

		boost::mutex::scoped_lock lock(m_mutex);
		m_not_full.wait(lock, boost::bind(&bounded_buffer<value_type>::is_ready_to_accept, this));
		if (is_not_full() && m_accept)
		{
			m_container.push_front(item);
			++m_unread;
			lock.unlock();
			m_not_empty.notify_one();
		}
		else
		{
			// check and remove if explicit unlock is required
			lock.unlock();
		}
	}

	bool try_push(typename boost::call_traits<value_type>::param_type item)
	{
		boost::mutex::scoped_lock lock(m_mutex);
		if (is_not_full() && m_accept)
		{
			m_container.push_front(item);
			++m_unread;
			lock.unlock();
			m_not_empty.notify_one();
			return true;
		}
		else {
			lock.unlock();
			return false;
		}
	}

	bool isFull() {
		bool iret = false;
		boost::mutex::scoped_lock lock(m_mutex);
		iret = !is_not_full();
		lock.unlock();
		return iret;
	}
	value_type pop() {
		boost::mutex::scoped_lock lock(m_mutex);
		m_not_empty.wait(lock, boost::bind(&bounded_buffer<value_type>::is_not_empty, this));
		--m_unread;
		value_type ret = m_container.back();
		m_container.pop_back();
		lock.unlock();
		m_not_full.notify_one();
		return ret;
	}

	value_type try_pop() {
		boost::mutex::scoped_lock lock(m_mutex);
		if (is_not_empty())
		{
			--m_unread;
			value_type ret = m_container.back();
			m_container.pop_back();
			lock.unlock();
			m_not_full.notify_one();
			return ret;
		}
		else {
			lock.unlock();
			return value_type();//empty container
		}
	}
	void clear() {
		boost::mutex::scoped_lock lock(m_mutex);
		m_container.clear();
		m_unread = 0;
		m_accept = false;
		m_not_full.notify_one();

		lock.unlock();
	}

	void accept() {
		boost::mutex::scoped_lock lock(m_mutex);
		m_accept = true;
		lock.unlock();
	}

	size_t size()
	{
		boost::mutex::scoped_lock lock(m_mutex);
		return m_container.size();
	}


private:
	bounded_buffer(const bounded_buffer&);              // Disabled copy constructor.
	bounded_buffer& operator = (const bounded_buffer&); // Disabled assign operator.

	bool is_not_empty() const { return m_unread > 0; }
	bool is_not_full() const { return m_unread < m_capacity; }
	bool is_ready_to_accept() const
	{
		return ((m_unread < m_capacity) || !m_accept);
	}

	bool m_accept;
	size_type m_unread;
	size_type m_capacity;
	container_type m_container;
	boost::mutex m_mutex;
	boost::condition m_not_empty;
	boost::condition m_not_full;

	friend class NonBlockingAllOrNonePushStrategy;

	void acquireLock()
	{
		m_mutex.lock();
	}

	void releaseLock()
	{
		m_mutex.unlock();
	}

	// to be used by QuePushStrategy Only
	void pushUnsafeForQuePushStrategy(typename boost::call_traits<value_type>::param_type item)
	{
		m_container.push_front(item);
		++m_unread;
		m_mutex.unlock();
		m_not_empty.notify_one();
	}
};
